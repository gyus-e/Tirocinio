import os
from src.cag.cag_prompt import build_cag_prompt
from ..ModelManager import ModelManager
from .cag import get_kv_cache, save_cache

from config import MODEL_NAME, CACHE_PATH

def init_cag_settings():
    model = ModelManager.get_model(MODEL_NAME)
    tokenizer = ModelManager.get_tokenizer(MODEL_NAME)

    system_prompt = build_cag_prompt()

    if not os.path.exists(CACHE_PATH):
        print("Cache not found, generating new cache...")
        my_cache = get_kv_cache(model, tokenizer, system_prompt)
        save_cache(my_cache)
        print("Cache saved successfully.")
 