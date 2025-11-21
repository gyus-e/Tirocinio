import chromadb
import logging
from llama_index.vector_stores.chroma import ChromaVectorStore
from config import chroma_host, chroma_port, chroma_collection_name as collection_name

db = (
    chromadb.HttpClient(host=chroma_host, port=chroma_port)
    if chroma_host is not None
    and chroma_port is not None
    and chroma_host != ""
    and chroma_port != ""
    else chromadb.EphemeralClient()
)
logging.debug("ChromaDB Client initialized.")


chroma_collection = db.get_or_create_collection(collection_name)
logging.debug(f"ChromaDB Collection {collection_name} ready.")


vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
logging.debug("VectorStore initialized.")
