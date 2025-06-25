import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM

from hf_token import HF_TOKEN
from params import (
    MODEL_NAME,
    EMBED_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TEMPERATURE,
    REQUEST_TIMEOUT,
    CONTEXT_WINDOW
)


def load_model(model_name: str = MODEL_NAME) :
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        token=HF_TOKEN, 
        trust_remote_code=True
    )

    return model, tokenizer

