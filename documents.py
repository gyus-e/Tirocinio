from llama_index.core import SimpleDirectoryReader
import json
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

documents = SimpleDirectoryReader("documents").load_data()
json_data = [json.loads(doc.text.strip()) for doc in documents]
