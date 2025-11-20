import torch
import os
from transformers.cache_utils import DynamicCache
from config import cag_system_prompt, cag_answer_instruction, kv_cache_path
from documents import doc_text


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
    kv = DynamicCache()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=kv,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )
    return outputs.past_key_values


def prepare_kvcache(
    model,
    tokenizer,
    documents,
    system_prompt: str,
    answer_instruction: str,
) -> DynamicCache:
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

    prompt = f"""
    <|start_header_id|>system<|end_header_id|>
    {system_prompt}
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Context information is below.
    ------------------------------------------------
    {documents}
    ------------------------------------------------
    {answer_instruction}
    Question:
    """

    kv = preprocess_knowledge(model, tokenizer, prompt)

    print("CAG Knowledge processed")
    return kv


def clean_up(kv: DynamicCache, origin_len: int):
    """
    Truncate the KV Cache to the original length.
    """
    for i in range(len(kv)):
        keys, values = kv[i]
        kv.layers[i].keys = keys[..., :origin_len, :] if keys is not None else None
        kv.layers[i].values = (
            values[..., :origin_len, :] if values is not None else None
        )


def generate(
    model,
    input_ids: torch.Tensor,
    kv: DynamicCache,
    max_new_tokens: int = 300,
) -> torch.Tensor:
    """
    Generate text with greedy decoding.

    Args:
        model: HuggingFace model with automatic device mapping
        input_ids: Input token ids
        kv: KV Cache for knowledge
        max_new_tokens: Maximum new tokens to generate
    """

    embed_device = model.model.embed_tokens.weight.device

    origin_ids = input_ids
    input_ids = input_ids.to(embed_device)

    output_ids = input_ids.clone()
    next_token = input_ids

    with torch.no_grad():
        # idea per evitare di pulire la kv:
        # usare una copia temporanea della kv
        # temp_kv = kv
        for _ in range(max_new_tokens):
            outputs = model(input_ids=next_token, past_key_values=kv, use_cache=True)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1).unsqueeze(-1)
            next_token = next_token.to(embed_device)

            kv = outputs.past_key_values

            output_ids = torch.cat([output_ids, next_token], dim=1)

            if (next_token.item() in model.config.eos_token_id) and (_ > 0):
                break

    return output_ids[:, origin_ids.shape[-1] :]


def get_or_create_kv_cache(
    model, tokenizer, kv_cache_path=kv_cache_path
) -> DynamicCache:
    if os.path.exists(kv_cache_path):
        kv = torch.load(kv_cache_path, weights_only=True)
        print("KV Cache loaded from disk.")
    else:
        kv = prepare_kvcache(
            model,
            tokenizer,
            documents=doc_text,
            system_prompt=cag_system_prompt,
            answer_instruction=cag_answer_instruction,
        )
        torch.save(kv, kv_cache_path)
        print("KV Cache created and saved to disk.")

    return kv


def get_kv_len(kv: DynamicCache) -> int:
    if len(kv) == 0:
        return 0

    keys, _ = kv[0]
    return keys.shape[2] if keys is not None else 0


def run_cag(model, tokenizer, kv: DynamicCache, kv_len: int, query: str):
    clean_up(kv, kv_len)
    input_ids = tokenizer.encode(query, return_tensors="pt").to(model.device)
    output = generate(model, input_ids, kv)
    return tokenizer.decode(output[0], skip_special_tokens=True, temperature=None)
