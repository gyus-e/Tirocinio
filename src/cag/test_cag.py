import torch

from params import CACHE_PATH
from .CAGModelManager import CAGModelManager
from ..test_questions import questions, delimiter
from .cag import get_answer, clean_up

def test_cag() -> None:
    model = CAGModelManager.get_model()
    tokenizer = CAGModelManager.get_tokenizer()
    torch_device = CAGModelManager.get_torch_device()

    for question in questions:
        loaded_cache = torch.load(CACHE_PATH)
        
        print("Q:", question)
        answer = get_answer(question, tokenizer, model, torch_device, loaded_cache)
        print("CAG:", answer)
        print(delimiter)

        clean_up(loaded_cache)

