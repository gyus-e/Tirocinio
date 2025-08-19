import torch
import bitsandbytes as bnb
from transformers.cache_utils import DynamicCache
import os


def preprocess_knowledge(model, tokenizer, prompt: str) -> DynamicCache:
    """
    Prepare knowledge kv cache for CAG by passing the prompt to the model.
    Args:
        model: HuggingFace model with automatic device mapping
        tokenizer: HuggingFace tokenizer
        prompt: The knowledge to preprocess, which is basically a prompt

    Returns:
        DynamicCache: KV Cache
    """

    # check which device is used. This depends on the chosen model.
    embed_device = model.model.embed_tokens.weight.device

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(embed_device)
    past_key_values = DynamicCache()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )
    return outputs.past_key_values


def prepare_kvcache(
    model,
    tokenizer,
    documents,
    system_prompt="You are a medical assistant for giving short answers based on given reports.",
    answer_instruction="Answer the question with a super short answer.",
):
    """
    Prepares kv cache for CAG by constructing a prompt with the content of the documents.
    The structure of this prompt, including any specific instructions or special tokens,
    is crucial and depends heavily on the chosen model.
    The size of the documents is also an important factor to consider,
    as it can overflow the context window of the model or saturate the memory of the machine that runs the program.
    Args:
        model: HuggingFace model with automatic device mapping
        tokenizer: HuggingFace tokenizer
        documents: any object that can be converted to a string, containing the documents contents
        system_prompt: The general instructions for the model
        answer_instruction: Other instructions for answering questions

    Returns:
        DynamicCache: KV Cache
    """

    knowledges = f"""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    {system_prompt}<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Context information is below.
    ------------------------------------------------
    {documents}
    ------------------------------------------------
    {answer_instruction}
    Question:
    """
    kv = preprocess_knowledge(model, tokenizer, knowledges)
    kv_len = kv.key_cache[0].shape[-2]
    print("kvlen: ", kv_len)
    return kv, kv_len


def write_kv_cache(kv: DynamicCache, path: str):
    """
    Write the KV Cache to a file.
    """
    torch.save(kv, path)


def read_kv_cache(path: str):
    """
    Read the KV Cache from a file.
    """
    kv = torch.load(path, weights_only=True)
    kv_len = kv.key_cache[0].shape[-2]
    return kv, kv_len


def clean_up(kv: DynamicCache, origin_len: int):
    """
    Truncate the KV Cache to the original length.
    """
    for i in range(len(kv.key_cache)):
        kv.key_cache[i] = kv.key_cache[i][:, :, :origin_len, :]
        kv.value_cache[i] = kv.value_cache[i][:, :, :origin_len, :]


def generate(
    model,
    input_ids: torch.Tensor,
    past_key_values: DynamicCache,
    max_new_tokens: int = 300,
) -> torch.Tensor:
    """
    Generate text with greedy decoding.

    Args:
        model: HuggingFace model with automatic device mapping
        input_ids: Input token ids
        past_key_values: KV Cache for knowledge
        max_new_tokens: Maximum new tokens to generate
    """

    embed_device = model.model.embed_tokens.weight.device

    origin_ids = input_ids
    input_ids = input_ids.to(embed_device)

    output_ids = input_ids.clone()
    next_token = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=next_token, past_key_values=past_key_values, use_cache=True
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1).unsqueeze(-1)
            next_token = next_token.to(embed_device)

            past_key_values = outputs.past_key_values

            output_ids = torch.cat([output_ids, next_token], dim=1)

            if (next_token.item() in model.config.eos_token_id) and (_ > 0):
                break
    return output_ids[:, origin_ids.shape[-1] :]


def get_or_create_kv_cache(model, tokenizer, kv_cache_path: str):
    from documents import json_data

    if os.path.exists(kv_cache_path):
        knowledge_cache, kv_len = read_kv_cache(kv_cache_path)
    else:
        knowledge_cache, kv_len = prepare_kvcache(model, tokenizer, documents=json_data)
        write_kv_cache(knowledge_cache, kv_cache_path)

    return knowledge_cache, kv_len


def run_cag(model, tokenizer, knowledge_cache: DynamicCache, kv_len: int, query: str):
    from cag import clean_up, generate

    clean_up(knowledge_cache, kv_len)
    input_ids = tokenizer.encode(query, return_tensors="pt").to(model.device)
    output = generate(model, input_ids, knowledge_cache)
    return tokenizer.decode(output[0], skip_special_tokens=True, temperature=None)
