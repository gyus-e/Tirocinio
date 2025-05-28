import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from cag import get_kv_cache, load_llm
from prompt import build_system_prompt
from test import test

def main():
    model, tokenizer, _ = load_llm()
    system_prompt = build_system_prompt()
    my_cache, device = get_kv_cache(model, tokenizer, system_prompt)
    origin_len = my_cache.key_cache[0].shape[-2]
    print("KV cache built.")

    cache_dir="cag_cache"
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(my_cache, os.path.join(cache_dir, "my_knowledge.cache"))

    test(model, tokenizer, device, my_cache, origin_len)


if __name__ == "__main__":
    main()

