import torch
from config import CACHE_PATH
from test_questions import questions, delimiter
from utils import ModelConfiguration
from .cag import get_answer, clean_up_cache
from . import model_configuration


def test_cag() -> None:
    model = model_configuration.model()
    tokenizer = model_configuration.tokenizer()
    torch_device = ModelConfiguration.torch_device()

    for question in questions:
        loaded_cache = torch.load(CACHE_PATH)

        print("Q:", question)
        answer = get_answer(question, tokenizer, model, torch_device, loaded_cache)
        print("CAG:", answer)
        print(delimiter)

        clean_up_cache(loaded_cache)
