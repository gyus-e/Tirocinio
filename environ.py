import os

# Must be set in the environment
HF_TOKEN = os.environ.get("HF_TOKEN", None)
HF_HOME = os.environ.get("HF_HOME", None)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)

# Safe to leave the default values
DOCUMENTS_DIR =  os.environ.get("DOCUMENTS_DIR", "_documents/test")
STORAGE = os.environ.get("STORAGE", "_storage")
CACHE_NAME = os.environ.get("CACHE_NAME", "cag.cache")
EMBED_MODEL_DIR = os.environ.get("EMBED_MODEL_DIR", "embed-models")

# Obtained from the previous variables
CACHE_PATH = os.path.join(STORAGE, CACHE_NAME)
VECTOR_STORE_DIR = os.path.join(STORAGE, "vector_store")