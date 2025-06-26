import os
from src.cag.cag_prompt import build_cag_prompt
from ..ModelConfiguration import ModelConfiguration
from .cag import create_kv_cache, save_cache
from config import MODEL_NAME, CACHE_PATH


def init_cag_settings(model_name=MODEL_NAME) -> ModelConfiguration:
    model_configuration = ModelConfiguration(model_name)
    model = model_configuration.model()
    tokenizer = model_configuration.tokenizer()

    system_prompt = build_cag_prompt()

    if not os.path.exists(CACHE_PATH):
        print("Cache not found, generating new cache...")
        my_cache = create_kv_cache(model, tokenizer, system_prompt)
        save_cache(my_cache)
        print("Cache saved successfully.")

    return model_configuration
