import os

DOCUMENTS_DIR = "_documents/test"
STORAGE = "_storage"
CACHE_NAME = "cag.cache"
CACHE_PATH = os.path.join(STORAGE, CACHE_NAME)
VECTOR_STORE_DIR = os.path.join(STORAGE, "vector_store")
EMBED_MODEL_DIR = "embed-models"

# User parameters

DO_CAG = True
DO_RAG = True

SYSTEM_PROMPT = """
    You are an assistant who provides concise factual answers.
    """

# LLM parameters
# Note: only MODEL_NAME is used for CAG, the rest are only used for RAG

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
CONTEXT_WINDOW = 8192
TEMPERATURE = 0.1
TOP_K = 50
TOP_P = 0.95


# RAG Parameters

EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 256
