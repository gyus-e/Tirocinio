import os 


SYSTEM_PROMPT = "You are an assistant who provides concise factual answers based on the context provided."
DOCUMENTS_DIR = "documents"


# LLM Parameters

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
TEMPERATURE = 0.25
REQUEST_TIMEOUT = 360.0
CONTEXT_WINDOW = 4096


# RAG Parameters

EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
VECTOR_STORE_DIR = "storage/vector_store"


# CAG Parameters

CACHE_DIR = "storage"
CACHE_NAME = "cag.cache"
CACHE_PATH = os.path.join(CACHE_DIR, CACHE_NAME)