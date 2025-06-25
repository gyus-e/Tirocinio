import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from llama_index.core import Settings

#FOR TESTING
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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


def init_llama_index_settings(model, tokenizer) -> None:
    # TODO: wrap the model and the tokenizer in a llama-index compatible format.
    # Settings.llm = HuggingFaceLLM(
    #     model=model,
    #     tokenizer=tokenizer,
    #     context_window=CONTEXT_WINDOW,
    #     device_map="auto",
    # )

    # for testing
    Settings.llm = OpenAI(
        model="gpt-3.5-turbo"
    )

    # Leave to default to use tiktoken, which is compatible with OpenAI models.
    # Settings.tokenizer = tokenizer

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
    )

    # Settings.chunk_size = CHUNK_SIZE
    # Settings.chunk_overlap = CHUNK_OVERLAP
