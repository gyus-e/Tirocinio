import os
import asyncio
import torch
from transformers.cache_utils import DynamicCache
from params import MODEL_NAME, CACHE_PATH
from src.prompt import build_system_prompt
from src.settings import load_model, init_llama_index_settings
from src.cag import get_device, get_kv_cache, save_cache
from src.test import test_cag, test_rag


async def main():
    torch.serialization.add_safe_globals([DynamicCache])
    model, tokenizer = load_model(MODEL_NAME)

    run_cag_test(model, tokenizer)
    await run_rag_test(model, tokenizer)
    

def run_cag_test(model, tokenizer):
    device = get_device(model)
    system_prompt = build_system_prompt()

    if not os.path.exists(CACHE_PATH):
        print("Cache not found, generating new cache...")
        my_cache = get_kv_cache(model, tokenizer, system_prompt)
        save_cache(my_cache)
        print("Cache saved successfully.")
 
    test_cag(model, tokenizer, device)


async def run_rag_test(model, tokenizer):
    init_llama_index_settings(model, tokenizer)
    await test_rag()


if __name__ == "__main__":
    asyncio.run(main())
