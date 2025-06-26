import os

SYSTEM_PROMPT = """You are an assistant who provides concise factual answers based on the context provided."""
DOCUMENTS_DIR = "documents"


# LLM Parameters
# Note: only MODEL_NAME is used for CAG, the rest are only used for RAG

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
CONTEXT_WINDOW = 3900
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.95


# RAG Parameters

EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
VECTOR_STORE_DIR = "storage/vector_store"


# CAG Parameters

CACHE_DIR = "storage"
CACHE_NAME = "cag.cache"
CACHE_PATH = os.path.join(CACHE_DIR, CACHE_NAME)
