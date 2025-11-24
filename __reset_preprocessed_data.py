import os
import chromadb
from config import chroma_host, chroma_port, kv_cache_path

db = chromadb.HttpClient(host=chroma_host, port=chroma_port)
for collection in db.list_collections():
    db.delete_collection(name=collection.name)

if os.path.exists(kv_cache_path.split("/")[1]):
    for file in os.listdir(f"{kv_cache_path.split('/')[1]}/"):
        os.remove(f"{kv_cache_path.split('/')[1]}/{file}")
