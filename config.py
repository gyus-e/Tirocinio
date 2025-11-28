MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
# MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
# MODEL_ID = "google/gemma-3-1b-it"
# MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

USE_4BIT_QUANTIZATION = True
MAX_NEW_TOKENS = 512

DOCUMENTS_DIR = "./documents"


## RAG configuration settings

# EMBED_MODEL_ID = "BAAI/bge-m3"
# EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MODEL_ID = "nickprock/sentence-bert-base-italian-uncased"
# EMBED_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"

EMBED_MODEL_PATH = "./models/embedding-model"

MAX_ITERATIONS = 10

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
CHROMA_COLLECTION_NAME = f"{EMBED_MODEL_ID.split('/')[1]}_embeddings"

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 256
RETRIEVE_TOP_K = 5

TEMPERATURE = 0.2
GENERATE_TOP_K = 40
GENERATE_TOP_P = 0.8
REPETITION_PENALITY = 1.1

RAG_SYSTEM_PROMPT = """
    Hai a disposizione documenti relativi al catalogo della Biblioteca Pontaniana di Napoli.
    Usa la funzione "search_documents" una sola volta per cercare informazioni utili.
    Dopo aver usato "search_documents", rispondi immediatamente utilizzando le informazioni trovate.
    Se non hai trovato informazioni rilevanti, rispondi con "Non lo so" e termina la risposta.
""".strip()


## CAG configuration settings

KV_CACHE_PATH = f"./kv_cache/kv_cache_{MODEL_ID.split('/')[1]}.pt"
CAG_SYSTEM_PROMPT = """
    Rispondi allo User Prompt utilizzando le informazioni che ti sono fornite.
    Se non sono sufficienti a rispondere, rispondi con "Non lo so" e termina la risposta.
""".strip()
