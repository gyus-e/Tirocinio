from llama_index.core import SimpleDirectoryReader
import sys
import io
from config import documents_dir

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

documents = SimpleDirectoryReader(documents_dir).load_data()
doc_text = [doc.text.strip() for doc in documents]
print("Documents loaded:", len(documents))
