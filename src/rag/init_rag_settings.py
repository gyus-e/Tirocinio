from llama_index.core import Settings
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from ..environ import OPENAI_API_KEY, HF_TOKEN, HF_HOME
from ..ModelManager import ModelManager
from config import (
    MODEL_NAME,
    EMBED_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CONTEXT_WINDOW,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    CACHE_DIR,
)


def init_rag_settings(model_name=MODEL_NAME, embed_model_name=EMBED_MODEL_NAME) -> None:

    # model = ModelManager.get_model(model_name)
    # tokenizer = ModelManager.get_tokenizer(model_name)
    # pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Settings.llm = HuggingFaceLLM(
    #     model_name=model_name,
    #     tokenizer_name=model_name,
    #     # model=model,
    #     # tokenizer=tokenizer,
    #     context_window=CONTEXT_WINDOW,
    #     # generate_kwargs={
    #     #     "temperature": TEMPERATURE if TEMPERATURE>0 else 0.1,
    #     #     "do_sample": True if TEMPERATURE==0 else False,
    #     #     "top_k": TOP_K,
    #     #     "top_p": TOP_P,
    #     #     "pad_token_id": pad_token_id,
    #     # },
    # )

    Settings.llm = HuggingFaceInferenceAPI(
        model=model_name,
        token=HF_TOKEN,
    )

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        cache_folder=HF_HOME,
    )

    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP
