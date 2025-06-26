import os

DOCUMENTS_DIR = "documents"


# It is recommended to clarify the context of the documents in the system prompt.

SYSTEM_PROMPT = """
    You must answer the user's questions with information extracted from a set of documents provided.
    Do not try to answer questions that are not related to the documents.
    The subject of the documents is the user's best friend.
    """


# LLM Parameters
# Note: only MODEL_NAME is used for CAG, the rest are only used for RAG

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
CONTEXT_WINDOW = 8192
TEMPERATURE = 0.1
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
