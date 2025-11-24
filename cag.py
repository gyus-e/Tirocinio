import logging
import torch
import os
from transformers.cache_utils import DynamicCache, DynamicLayer
from model import model, tokenizer, device
from config import (
    cag_system_prompt,
    cag_answer_instruction,
    max_new_tokens,
)

torch.serialization.add_safe_globals([DynamicCache, DynamicLayer])


def get_or_create_kv_cache(kv_cache_path) -> DynamicCache:

    if os.path.exists(kv_cache_path):
        kv = torch.load(kv_cache_path, weights_only=True)
        logging.debug("KV Cache loaded from disk.")
    else:
        from documents import documents  # Load the documents only if they are needed

        prompt = __build_prompt(
            documents,
            cag_system_prompt,
            cag_answer_instruction,
        )

        kv = __prepare_kvcache(
            model,
            tokenizer,
            prompt,
        )
        torch.save(kv, kv_cache_path)
        logging.debug("KV Cache created and saved to disk.")

    return kv


def get_kv_len(kv: DynamicCache) -> int:
    if len(kv) == 0:
        return 0

    keys, _ = kv[0]
    return keys.shape[2] if keys is not None else 0


def run_cag(kv: DynamicCache, query: str):
    input_ids = tokenizer.encode(query, return_tensors="pt").to(model.device)
    output = __generate(model, input_ids, kv, max_new_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True, temperature=None)


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


def __build_prompt(documents, system_prompt: str, answer_instruction: str = "") -> str:
    return f"""
    <|start_header_id|>system<|end_header_id|>
    {system_prompt}
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Contesto:
    ------------------------------------------------
    {documents}
    ------------------------------------------------
    {answer_instruction}
    Domanda:
    """


def __preprocess_knowledge(model, tokenizer, prompt: str) -> DynamicCache:
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
    embed_device = (
        "cuda" if device == "cuda" else model.model.embed_tokens.weight.device
    )

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


def __prepare_kvcache(
    model,
    tokenizer,
    prompt: str,
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
        prompt: The knowledge to preprocess, which is basically a prompt

    Returns:
        DynamicCache: KV Cache
    """

    kv = __preprocess_knowledge(model, tokenizer, prompt)

    logging.debug("CAG Knowledge processed")
    return kv


def __generate(
    model,
    input_ids: torch.Tensor,
    kv: DynamicCache,
    max_new_tokens: int,
) -> torch.Tensor:
    """
    Generate text with greedy decoding.

    Args:
        model: HuggingFace model with automatic device mapping
        input_ids: Input token ids
        kv: KV Cache for knowledge
        max_new_tokens: Maximum new tokens to generate
    """

    embed_device = (
        "cuda" if device == "cuda" else model.model.embed_tokens.weight.device
    )

    origin_ids = input_ids
    input_ids = input_ids.to(embed_device)

    output_ids = input_ids.clone()
    next_token = input_ids

    with torch.no_grad():
        # idea per evitare di pulire la kv: usare una copia temporanea (ma questo credo sia un riferimento)
        temp_kv = kv

        eos_ids = (
            [model.config.eos_token_id]
            if isinstance(model.config.eos_token_id, int)
            else model.config.eos_token_id
        )

        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=next_token,
                past_key_values=temp_kv,
                use_cache=True,
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1).unsqueeze(-1)
            next_token = next_token.to(embed_device)

            temp_kv = outputs.past_key_values

            output_ids = torch.cat([output_ids, next_token], dim=1)

            if (next_token.item() in eos_ids) and (_ > 0):
                break

    return output_ids[:, origin_ids.shape[-1] :]
