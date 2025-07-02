import os
from environ import HF_HOME
from config import EMBED_MODEL_DIR

def list_models() -> list[str]:
    models_list = os.listdir(f"{HF_HOME}/hub")
    print("Available models:", *[(f"{i}: [{model_name}]") for i, model_name in enumerate(models_list) if not model_name.startswith('.')], sep="\n")
    return models_list

def list_embed_models() -> list[str]:
    embed_models_list = os.listdir(f"{HF_HOME}/{EMBED_MODEL_DIR}")
    print("Available embed models:", *[(f"{i}: [{model_name}]") for i, model_name in enumerate(embed_models_list) if not model_name.startswith('.')], sep="\n")
    return embed_models_list