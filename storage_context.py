import chromadb
import logging
from llama_index.vector_stores.chroma import ChromaVectorStore
from config import CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION_NAME as collection_name

DB = (
    chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    if CHROMA_HOST is not None
    and CHROMA_PORT is not None
    and CHROMA_HOST != ""
    and CHROMA_PORT != ""
    else chromadb.EphemeralClient()
)
logging.debug("ChromaDB Client initialized.")


CHROMA_COLLECTION = DB.get_or_create_collection(collection_name)
logging.debug(f"ChromaDB Collection {collection_name} ready.")


VECTOR_STORE = ChromaVectorStore(chroma_collection=CHROMA_COLLECTION)
logging.debug("VectorStore initialized.")
