import os
from environ import HF_HOME

def list_models() -> list[str]:
    models_list = os.listdir(f"{HF_HOME}/hub")
    print("Available models:", [model for model in models_list if not model.startswith('.')], sep="\n")
    return models_list
