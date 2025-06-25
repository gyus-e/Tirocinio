from llama_index.core import Document, SimpleDirectoryReader
from params import documents_dir

class DocumentsManager:
    _documents:list[Document]
    _initialized:bool = False


    @classmethod
    def get_documents(cls) -> list[Document]:
        if not cls._initialized:
            cls._load_documents()
        return cls._documents


    @classmethod
    def _load_documents(cls):
        print("Loading documents from directory...")

        cls._documents = SimpleDirectoryReader(input_dir=documents_dir).load_data()

        print(f"Loaded {len(cls._documents)} documents.")
        cls._initialized = True
