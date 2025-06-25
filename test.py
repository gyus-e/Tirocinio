from cag import get_answer, clean_up
from params import CACHE_PATH
import torch

def test(model, tokenizer, device):
    questions = [
        "Can you tell me what the context is about?",
        "What city does My best friend live in?",
        "What color are My best friend's eyes?",
        "What are My best friend's favorite writers?",
        "What is My best friend's cat called?",
        "Tell me about my best friend's family.",
        "What are the hobbies of my best friend's wife?"
    ]
    for question in questions:
        loaded_cache = torch.load(CACHE_PATH)
        print("Q:", question)
        answer = get_answer(question, tokenizer, model, device, loaded_cache)
        print("A:", answer)
        clean_up(loaded_cache)
