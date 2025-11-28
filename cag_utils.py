import logging
import torch
import os
from transformers.cache_utils import DynamicCache


def load_kv_cache(kv_cache_path: str) -> DynamicCache:
    """Load an existing KV Cache from disk at the specified path."""
    if os.path.exists(kv_cache_path):
        kv = torch.load(kv_cache_path, weights_only=True)
        logging.debug("KV Cache loaded from disk.")
        return kv
    else:
        raise FileNotFoundError(f"KV cache not found at {kv_cache_path}.")


def create_kv_cache(
    kv_cache_path: str,
    system_prompt: str,
    documents,
    model,
    tokenizer,
    embed_device,
) -> DynamicCache:
    """Create a new KV Cache and save it to disk at the specified path, overwriting any existing file."""
    if os.path.exists(kv_cache_path):
        os.remove(kv_cache_path)
        logging.debug("Existing KV Cache file removed.")
    prompt = __build_prompt(
        documents,
        system_prompt,
    )
    kv = __preprocess_knowledge(
        model,
        tokenizer,
        embed_device,
        prompt,
    )
    logging.debug("CAG Knowledge processed")
    torch.save(kv, kv_cache_path)
    logging.debug("KV Cache created and saved to disk.")
    return kv


def get_kv_len(kv: DynamicCache) -> int:
    """Get the length of the KV Cache."""
    if len(kv) == 0:
        return 0
    keys, _ = kv[0]
    return keys.shape[2] if keys is not None else 0


def generate_response(
    kv: DynamicCache,
    query: str,
    model,
    tokenizer,
    embed_device,
    max_new_tokens: int,
) -> str:
    """Generate a response using the cached knowledge and the provided query."""
    input_ids = tokenizer.encode(query, return_tensors="pt").to(model.device)
    output = __generate_response_from_input_ids(
        model, embed_device, input_ids, kv, max_new_tokens
    )
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


def __build_prompt(documents, system_prompt: str) -> str:
    return f"""
    System Prompt:
    {system_prompt.strip()}
    Context Information:
    {str(documents).strip()}
    User Prompt:
    """.strip()


def __preprocess_knowledge(model, tokenizer, embed_device, prompt: str) -> DynamicCache:
    """
    Prepare knowledge kv cache for CAG by passing the prompt to the model.
    Args:
        model: HuggingFace model with automatic device mapping
        tokenizer: HuggingFace tokenizer
        prompt: The knowledge to preprocess, which is basically a prompt

    Returns:
        DynamicCache: KV Cache
    """
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


def __generate_response_from_input_ids(
    model,
    embed_device,
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
    origin_ids = input_ids
    input_ids = input_ids.to(embed_device)

    output_ids = input_ids.clone()
    next_token = input_ids

    with torch.no_grad():
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
