from llama_index.core import SimpleDirectoryReader
from config import documents_dir

documents = SimpleDirectoryReader(documents_dir).load_data()
doc_text = [doc.text.strip() for doc in documents]
print("Documents loaded:", len(documents))
