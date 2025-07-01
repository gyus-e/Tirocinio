import os
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.indices.base import BaseIndex

from config import VECTOR_STORE_DIR
from utils import DocumentsManager


class IndexManager:
    _index: BaseIndex
    _initialized: bool = False

    @classmethod
    async def get_index(cls) -> BaseIndex:
        if not cls._initialized:
            await cls._initialize_index()
        return cls._index

    @classmethod
    async def _initialize_index(cls):

        if not os.path.exists(VECTOR_STORE_DIR):
            print("Creating index...")
            cls._create_index()

        else:
            print("Loading index from storage...")
            cls._load_index()

        cls._initialized = True

    @classmethod
    def _create_index(cls):
        documents = DocumentsManager.get_documents()
        cls._index = VectorStoreIndex.from_documents(documents)

        print("Index created. Persisting to storage...")
        cls._index.storage_context.persist(persist_dir=VECTOR_STORE_DIR)
        print("Index persisted to storage.")

    @classmethod
    def _load_index(cls):
        storage_context = StorageContext.from_defaults(persist_dir=VECTOR_STORE_DIR)
        cls._index = load_index_from_storage(storage_context)
        print("Index loaded from storage.")
