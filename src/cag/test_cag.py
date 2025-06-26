import torch

from config import CACHE_PATH
from ..test_questions import questions, delimiter
from ..ModelManager import ModelManager
from .cag import get_answer, clean_up


def test_cag() -> None:
    model = ModelManager.get_model()
    tokenizer = ModelManager.get_tokenizer()
    torch_device = ModelManager.get_torch_device()

    for question in questions:
        loaded_cache = torch.load(CACHE_PATH)
        
        print("Q:", question)
        answer = get_answer(question, tokenizer, model, torch_device, loaded_cache)
        print("CAG:", answer)
        print(delimiter)

        clean_up(loaded_cache)

