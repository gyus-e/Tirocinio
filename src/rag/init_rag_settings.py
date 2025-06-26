import os
import torch
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from ..environ import OPENAI_API_KEY, HF_TOKEN, HF_HOME
from ..ModelConfiguration import ModelConfiguration
from config import (
    MODEL_NAME,
    EMBED_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CONTEXT_WINDOW,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)


def init_rag_settings(
    model_name=MODEL_NAME, embed_model_name=EMBED_MODEL_NAME
) -> ModelConfiguration:

    model_configuration = ModelConfiguration(model_name)
    model = model_configuration.model()
    tokenizer = model_configuration.tokenizer()

    temperature = (TEMPERATURE if TEMPERATURE > 0 else 0.1,)
    do_sample = True if TEMPERATURE == 0 else False
    kwargs = {
        "temperature": temperature,
        "do_sample": do_sample,
        "top_k": TOP_K,
        "top_p": TOP_P,
    }

    Settings.llm = HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer,
        # generate_kwargs=kwargs,
    )

    # Settings.llm = HuggingFaceInferenceAPI(
    #     model=model_name,
    #     token=HF_TOKEN,
    # )

    # Settings.llm = OpenAI(
    #     model="gpt-3.5-turbo",
    #     api_key=OPENAI_API_KEY,
    # )

    # Settings.tokenizer = tokenizer

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=embed_model_name,
        cache_folder=os.path.join(HF_HOME, "hub") if HF_HOME else None,
    )

    # Settings.chunk_size = CHUNK_SIZE
    # Settings.chunk_overlap = CHUNK_OVERLAP

    return model_configuration
