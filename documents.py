import logging
from llama_index.core import SimpleDirectoryReader
from config import documents_dir

documents = SimpleDirectoryReader(documents_dir).load_data()
logging.debug("Documents loaded:", len(documents))
