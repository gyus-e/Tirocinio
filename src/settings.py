import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from hf_token import HF_TOKEN
from params import (
    MODEL_NAME,
    EMBED_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def load_model(model_name: str = MODEL_NAME) :
    model = AutoModelForCausalLM.from_pretrained(

        pretrained_model_name_or_path=model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
    
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name, token=HF_TOKEN, trust_remote_code=True
    )

    return model, tokenizer


def init_settings() -> None:

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
    )

    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP
