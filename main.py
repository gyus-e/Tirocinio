import os
import torch
from transformers.cache_utils import DynamicCache
from cag import load_llm, get_device, get_kv_cache, save_cache
from prompt import build_system_prompt
from llm_model import MODEL_NAME
from cache_params import CACHE_PATH
from test import test


def main():
    torch.serialization.add_safe_globals([DynamicCache])
    system_prompt = build_system_prompt()
    model, tokenizer, _ = load_llm(MODEL_NAME)
    device = get_device(model)

    if not os.path.exists(CACHE_PATH):
        print("Cache not found, generating new cache...")
        my_cache = get_kv_cache(model, tokenizer, system_prompt)
        save_cache(my_cache)
        print("Cache saved successfully.")
 
    test(model, tokenizer, device)


if __name__ == "__main__":
    main()
