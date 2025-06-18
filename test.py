from cag import get_answer, clean_up
from cache_params import CACHE_PATH
import torch

def test(model, tokenizer, device):
    questions = [
        "Can you tell me what the context is about?",
        "What is My best friend's father's job?",
        "What city does My best friend live in?",
        "What color are My best friend's eyes?",
        "What are My best friend's favorite writers?",
        "What is My best friend's cat called?"
    ]
    for question in questions:
        loaded_cache = torch.load(CACHE_PATH)
        print("Q:", question)
        answer = get_answer(question, tokenizer, model, device, loaded_cache)
        print("A:", answer)
        clean_up(loaded_cache)
