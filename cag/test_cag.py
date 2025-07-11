import torch
from transformers.cache_utils import DynamicCache
from environ import CACHE_PATH
from test_questions import questions, delimiter
from utils import ModelConfiguration
from .cag import get_answer, clean_up_cache
from . import initialize_settings


def test_cag() -> None:
    torch.serialization.add_safe_globals([DynamicCache])
    model_configuration = initialize_settings()
    model = model_configuration.model() if model_configuration else None
    tokenizer = model_configuration.tokenizer() if model_configuration else None
    torch_device = ModelConfiguration.torch_device()

    for question in questions:
        loaded_cache = torch.load(CACHE_PATH)

        print("Q:", question)
        answer = get_answer(question, tokenizer, model, torch_device, loaded_cache)
        print("CAG:", answer)
        print(delimiter)

        clean_up_cache(loaded_cache)
