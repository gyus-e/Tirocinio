import logging
from llama_index.core import SimpleDirectoryReader
from config import DOCUMENTS_DIR

DOCUMENTS = SimpleDirectoryReader(DOCUMENTS_DIR).load_data()
logging.debug("Documents loaded: %d", len(DOCUMENTS))
