import os
from environ import HF_HOME
from config import EMBED_MODEL_DIR

def get_models_list() -> list[str]:
    models_list = os.listdir(f"{HF_HOME}/hub") if os.path.exists(f"{HF_HOME}/hub") else []
    if not models_list:
        return []
    return [_parse_model_name(model_name) for model_name in models_list if not model_name.startswith('.')]

def get_embed_models_list() -> list[str]:
    embed_models_list = os.listdir(f"{HF_HOME}/{EMBED_MODEL_DIR}") if os.path.exists(f"{HF_HOME}/{EMBED_MODEL_DIR}") else []
    if not embed_models_list:
        return []
    return [_parse_model_name(model_name) for model_name in embed_models_list if not model_name.startswith('.')]

def _parse_model_name(model_name: str) -> str:
    return model_name.removeprefix("models--").replace("--", "/")