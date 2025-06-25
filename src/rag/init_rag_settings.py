from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers.models.auto.tokenization_auto import AutoTokenizer

#FOR TESTING
# from llama_index.llms.huggingface import HuggingFaceLLM
# from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

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


def init_rag_settings(model_name = MODEL_NAME) -> None:
    
    # Leave default tokenizer to use tiktoken, which is compatible with OpenAI models.
    # Settings.tokenizer = AutoTokenizer.from_pretrained(
    #     model_name, 
    #     token=HF_TOKEN, 
    #     trust_remote_code=True
    # )

    # Settings.llm = HuggingFaceLLM(
    #     model=model_name,
    #     tokenizer=Settings.tokenizer,
    #     context_window=CONTEXT_WINDOW,
    #     device_map="auto",
    # )

    # for testing
    Settings.llm = OpenAI(
        model="gpt-3.5-turbo"
    )

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
    )

    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP
