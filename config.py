import os

DOCUMENTS_DIR = "documents\test"

DO_CAG = True
DO_RAG = True

# It is recommended to specify the context of the documents in the system prompt.

# SYSTEM_PROMPT = """
#     Rispondi alle domande dell'utente utilizzando le informazioni contenute nei documenti.
#     L'argomento dei documenti è il catalogo della biblioteca pontaniana di Napoli.
#     """

SYSTEM_PROMPT = """
    Answer the user's questions using the information contained in the documents.
    The documents are about the user's best friend.
    """

# LLM Parameters
# Note: only MODEL_NAME is used for CAG, the rest are only used for RAG

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
CONTEXT_WINDOW = 8192
# CONTEXT_WINDOW = 32768
TEMPERATURE = 0.1
TOP_K = 50
TOP_P = 0.95


# RAG Parameters

EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 256
VECTOR_STORE_DIR = "storage/vector_store"


# CAG Parameters

CACHE_DIR = "storage"
CACHE_NAME = "cag.cache"
CACHE_PATH = os.path.join(CACHE_DIR, CACHE_NAME)
