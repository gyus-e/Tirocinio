from cag import get_answer
from cache_params import CACHE_PATH
import torch

def test(model, tokenizer, device):
    questions = [
        "Who is Roberto Anselmi?",
        "What is Roberto Anselmi's father's job?",
        "What city does Roberto Anselmi live in?",
        "What color are Roberto Anselmi's eyes?",
        "What are Roberto Anselmi's favorite writers?",
        "What is Roberto Anselmi's cat called?"
    ]
    for question in questions:
        loaded_cache = torch.load(CACHE_PATH)
        print("Q:", question)
        answer = get_answer(question, tokenizer, model, device, loaded_cache)
        print("A:", answer)
