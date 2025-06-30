import torch
from config import CACHE_PATH
from test_questions import questions, delimiter
from ..ModelConfiguration import ModelConfiguration
from .cag import get_answer, clean_up_cache


def run_cag(model_configuration: ModelConfiguration) -> None:
    model = model_configuration.model()
    tokenizer = model_configuration.tokenizer()
    torch_device = ModelConfiguration.torch_device()

    print("Press ctrl+C to exit.")
    while True:
        loaded_cache = torch.load(CACHE_PATH)

        question = input()
        answer = get_answer(question, tokenizer, model, torch_device, loaded_cache)
        print("CAG:", answer)
        print(delimiter)

        clean_up_cache(loaded_cache)
