import os
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from environ import OPENAI_API_KEY, HF_TOKEN, HF_HOME, EMBED_MODEL_DIR
from config import (
    MODEL_NAME,
    EMBED_MODEL_NAME,
    CONTEXT_WINDOW,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)

def initialize_settings():
    
    kwargs = {"temperature": TEMPERATURE} if TEMPERATURE > 0 else {"do_sample": True}
    kwargs = {"top_k": TOP_K} if TOP_K else kwargs
    kwargs = {"top_p": TOP_P} if TOP_P else kwargs

    Settings.llm = HuggingFaceLLM(
        model_name=MODEL_NAME,
        tokenizer_name=MODEL_NAME,
        context_window=CONTEXT_WINDOW,
        generate_kwargs=kwargs,
    )

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        cache_folder=os.path.join(HF_HOME, EMBED_MODEL_DIR) if HF_HOME else None,
    )

    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP

    # For testing, comment all the above and uncomment the following lines. Set up your OpenAI API key in the .env file.
    # Settings.llm = OpenAI(
    #     model="gpt-3.5-turbo",
    #     api_key=OPENAI_API_KEY,
    # )
