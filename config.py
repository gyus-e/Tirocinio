import os

DOCUMENTS_DIR = "documents/test"

DO_CAG = True
DO_RAG = True

SYSTEM_PROMPT = """
    You are an assistant who provides concise factual answers.
    """

# LLM Parameters
# Note: only MODEL_NAME is used for CAG, the rest are only used for RAG

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
CONTEXT_WINDOW = 8192
TEMPERATURE = 0.1
TOP_K = 50
TOP_P = 0.95


# RAG Parameters

EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE = 2048  # Optimized for Llama 3.2 - larger chunks for better context retention
CHUNK_OVERLAP = 512  # Increased overlap proportionally to maintain context continuity
VECTOR_STORE_DIR = "storage/vector_store"


# CAG Parameters

CACHE_DIR = "storage"
CACHE_NAME = "cag.cache"
CACHE_PATH = os.path.join(CACHE_DIR, CACHE_NAME)
