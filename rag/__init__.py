import os
import torch
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from environ import OPENAI_API_KEY, HF_TOKEN, HF_HOME
from utils import ModelConfiguration
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


# model_configuration = ModelConfiguration(MODEL_NAME)
# model = model_configuration.model()
# tokenizer = model_configuration.tokenizer()

# temperature = TEMPERATURE if TEMPERATURE > 0 else 0.1
# do_sample = True if TEMPERATURE == 0 else False
# kwargs = {
#     "temperature": temperature,
#     "do_sample": do_sample,
#     "top_k": TOP_K,
#     "top_p": TOP_P,
# }

Settings.llm = HuggingFaceLLM(
    model_name=MODEL_NAME,
    tokenizer_name=MODEL_NAME,
    context_window=CONTEXT_WINDOW,
    # model=model,
    # tokenizer=tokenizer,
    # generate_kwargs=kwargs,
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL_NAME,
    cache_folder=os.path.join(HF_HOME, "embed-models") if HF_HOME else None,
)

Settings.chunk_size = CHUNK_SIZE
Settings.chunk_overlap = CHUNK_OVERLAP

# For testing, comment all the above and uncomment the following lines. Set up your OpenAI API key in the .env file.
# Settings.llm = OpenAI(
#     model="gpt-3.5-turbo",
#     api_key=OPENAI_API_KEY,
# )
