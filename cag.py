import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from hf_token import hf_token as HF_TOKEN


def generate(model, input_ids, past_key_values, max_new_tokens: int = 50):
    """The generate function handles token-by-token generation with the cached knowledge using greedy decoding."""
    """Greedy decoding is a simple text generation method where, at each step, the token with the highest probability (maximum value in the logits) is selected as the next token."""
    """@param model: The LLM."""
    """@param input_ids: A tensor containing the tokenized input sequence."""
    """@param past_key_values: the core component of CAG: a cache of previously computed attention values used to speed up inference by avoiding recomputation."""
    """@param max_new_tokens: The maximum number of new tokens to generate."""

    device = get_device(model)


    origin_len = input_ids.shape[-1]
    input_ids = input_ids.to(device)
    output_ids = input_ids.clone()
    next_token = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = out.logits[:, -1, :]
            token = torch.argmax(logits, dim=-1, keepdim=True)
            output_ids = torch.cat([output_ids, token], dim=-1)
            past_key_values = out.past_key_values
            next_token = token.to(device)

            if model.config.eos_token_id is not None and token.item() == model.config.eos_token_id:
                break
    
    return output_ids[:, origin_len:]

def get_device(model):
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        # Mistral/Llama models
        device = model.model.embed_tokens.weight.device
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        # GPT-2 models
        device = model.transformer.wte.weight.device
    else:
        # Fallback to first parameter device
        device = next(model.parameters()).device
    print(f"Using device: {device}")
    return device


def get_kv_cache(model, tokenizer, prompt: str):
    """prepares a reusable key-value cache for a transformer model's attention mechanism."""
    """passes a prompt through the model once, creating a KV cache that records all the hidden states from each layer"""
    """@param model: the transformer model."""
    """@param tokenizer: the tokenizer to convert the prompt into the token IDs."""
    """@prompt: a string used as the prompt"""
    """@return: DynamicCache object containing the key-value cache."""

    device = get_device(model)
    print(f"Using device: {device}")

    # Tokenize the prompt and convert it into input IDs
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    print("Prompt tokenized.")

    # Initialize the DynamicCache object
    cache = DynamicCache()
    print("DynamicCache initialized.")

    # Perform forward pass through the model with caching enabled
    with torch.no_grad():
        _ = model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
        )

    print("KV cache created.")
    return cache, device


def load_llm():
    # model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    # model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    # model.to(device) # Removed: accelerate handles device mapping automatically
    print(f"Loaded {model_name}")

    return model, tokenizer, device_type
    

def clean_up(cache: DynamicCache, origin_len: int):
    """Cleans the key-value cache by removing unnecessary entries"""
    """Trims a DynamicCache object to match the original sequence length by removing additional tokens added during processing"""
    """For each layer of the cache, it slices both the key and value tensors to retain only the first origin_len tokens along the sequence dimension"""
    for i in range(len(cache.key_cache)):
        cache.key_cache[i] = cache.key_cache[i][:, :, origin_len:, :]
        cache.value_cache[i] = cache.value_cache[i][:, :, origin_len:, :]

