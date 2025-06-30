import os

DOCUMENTS_DIR = "documents/pontaniana"

# Set to True for CAG, False for RAG
CAG = False  


# It is recommended to clarify the context of the documents in the system prompt.

SYSTEM_PROMPT = """
    Rispondi alle domande dell'utente utilizzando le informazioni contenute nei documenti.
    L'argomento dei documenti è il catalogo della biblioteca pontaniana di Napoli.
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
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
VECTOR_STORE_DIR = "storage/vector_store"


# CAG Parameters

CACHE_DIR = "storage"
CACHE_NAME = "cag.cache"
CACHE_PATH = os.path.join(CACHE_DIR, CACHE_NAME)
