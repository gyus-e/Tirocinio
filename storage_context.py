import chromadb
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

ephemeral_client = True
db = chromadb.EphemeralClient() if ephemeral_client else chromadb.HttpClient()

try:
    chroma_collection = db.get_collection("quickstart")
    collection_exists = True
except Exception as e:
    chroma_collection = db.create_collection("quickstart")
    collection_exists = False

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
