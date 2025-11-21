from llama_index.core import SimpleDirectoryReader
from config import documents_dir

documents = SimpleDirectoryReader(documents_dir).load_data()
print("Documents loaded:", len(documents))
