import os

DOCUMENTS_DIR = "_documents/test"
STORAGE = "_storage"
CACHE_NAME = "cag.cache"
CACHE_PATH = os.path.join(STORAGE, CACHE_NAME)
VECTOR_STORE_DIR = os.path.join(STORAGE, "vector_store")

# User parameters

DO_CAG = True
DO_RAG = True

SYSTEM_PROMPT = """
    Answer the user's questions using the information contained in the documents.
    The documents are about the user's best friend.
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


# class Config:
#     do_cag: bool
#     do_rag: bool
#     system_prompt: str
#     model_name: str
#     __initialized: bool = False

#     @classmethod
#     def initialize(
#         cls,
#         do_cag: bool,
#         do_rag: bool,
#         system_prompt: str,
#         model_name: str,
#     ):
#         cls.do_cag = do_cag
#         cls.do_rag = do_rag
#         cls.system_prompt = system_prompt
#         cls.model_name = model_name
#         cls.__initialized = True


# class RagConfig:
#     context_window: int
#     temperature: float
#     top_k: int
#     top_p: float
#     __initialized: bool = False

#     @classmethod
#     def initialize(
#         cls,
#         context_window: int,
#         temperature: float,
#         top_k: int,
#         top_p: float,
#     ):
#         cls.context_window = context_window
#         cls.temperature = temperature
#         cls.top_k = top_k
#         cls.top_p = top_p
#         cls.__initialized = True
