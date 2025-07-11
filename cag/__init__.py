import os
from cag.cag_prompt import build_cag_prompt
from utils import ModelConfiguration
from config import MODEL_NAME
from environ import CACHE_PATH
from .cag import create_kv_cache, save_cache

def initialize_settings():
    model_configuration = ModelConfiguration(MODEL_NAME)
    model = model_configuration.model()
    tokenizer = model_configuration.tokenizer()

    system_prompt = build_cag_prompt()

    if not os.path.exists(CACHE_PATH):
        print("Cache not found, generating new cache...")
        my_cache = create_kv_cache(model, tokenizer, system_prompt)
        save_cache(my_cache)
        print("Cache saved successfully.")
    
    return model_configuration
