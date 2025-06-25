import os
from src.cag.cag_prompt import build_cag_prompt
from src.cag.CAGModelManager import CAGModelManager
from src.cag.cag import get_kv_cache, save_cache

from params import MODEL_NAME, CACHE_PATH

def init_cag_settings():
    model = CAGModelManager.get_model(MODEL_NAME)
    tokenizer = CAGModelManager.get_tokenizer(MODEL_NAME)

    system_prompt = build_cag_prompt()

    if not os.path.exists(CACHE_PATH):
        print("Cache not found, generating new cache...")
        my_cache = get_kv_cache(model, tokenizer, system_prompt)
        save_cache(my_cache)
        print("Cache saved successfully.")
 