import os
import chromadb

chroma_host = "localhost"
chroma_port = 8000
kv_cache_path = "../kv_cache"

db = chromadb.HttpClient(host=chroma_host, port=chroma_port)
for collection in db.list_collections():
    print(f"Deleting collection: {collection.name}")
    db.delete_collection(name=collection.name)

kv_cache_dir = "/".join(kv_cache_path.split("/")[:2])
if os.path.exists(kv_cache_dir):
    for file in os.listdir(f"{kv_cache_dir}/"):
        print(f"Removing file: {kv_cache_dir}/{file}")
        os.remove(f"{kv_cache_dir}/{file}")
