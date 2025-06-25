import os
import torch
from transformers.cache_utils import DynamicCache
from llama_index.core import Settings
from params import MODEL_NAME, CACHE_PATH
from src.prompt import build_system_prompt
from src.settings import load_model, init_settings
from src.cag import get_device, get_kv_cache, save_cache
from src.test import test


def main():
    torch.serialization.add_safe_globals([DynamicCache])
    model, tokenizer = load_model(MODEL_NAME)
    device = get_device(model)
    system_prompt = build_system_prompt()
    # init_settings() #only for RAG

    if not os.path.exists(CACHE_PATH):
        print("Cache not found, generating new cache...")
        my_cache = get_kv_cache(model, tokenizer, system_prompt)
        save_cache(my_cache)
        print("Cache saved successfully.")
 
    test(model, tokenizer, device)


if __name__ == "__main__":
    main()
