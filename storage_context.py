import chromadb
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from config import chroma_host, chroma_port, chroma_collection_name as collection_name

use_ephemeral_client = chroma_host is None or chroma_port is None

db = (
    chromadb.EphemeralClient()
    if chroma_host is None or chroma_port is None
    else chromadb.HttpClient(host=chroma_host, port=chroma_port)
)
print("ChromaDB Client initialized.")
if use_ephemeral_client:
    print("Using EphemeralClient.")

collection_already_exists = collection_name in [c.name for c in db.list_collections()]

chroma_collection = (
    db.get_collection(collection_name)
    if collection_already_exists
    else db.create_collection(collection_name)
)
print("ChromaDB Collection ready.")


vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
print("VectorStore initialized.")

storage_context = StorageContext.from_defaults(vector_store=vector_store)
print("StorageContext created.")
