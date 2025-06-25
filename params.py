import os 

# General Parameters

CAG = True
RAG = False
documents_dir = "documents" # Change this to your documents directory
llm_model = "llama3.2"
llm_base_url = "http://localhost:11434"
llm_temperature = 0.25
llm_request_timeout = 360.0
llm_context_window = 4096

# RAG Parameters

embed_model_name = "BAAI/bge-base-en-v1.5"
tokenizer_name = "meta-llama/Llama-3.2-3B-Instruct"
chunk_size = 512
chunk_overlap = 128
persist_dir = "storage/vector_store_pontaniana"

# CAG Parameters

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
CACHE_DIR = "cag_cache"
CACHE_NAME = "my_knowledge.cache"
CACHE_PATH = os.path.join(CACHE_DIR, CACHE_NAME)