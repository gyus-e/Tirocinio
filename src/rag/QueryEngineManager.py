from llama_index.core import Settings
from llama_index.core.base.base_query_engine import BaseQueryEngine
from .IndexManager import IndexManager

class QueryEngineManager:
    _query_engine: BaseQueryEngine
    _initialized: bool = False


    @classmethod
    async def get_query_engine(cls) -> BaseQueryEngine:
        if not cls._initialized:
            await cls._create_query_engine()
        return cls._query_engine


    @classmethod
    async def _create_query_engine(cls):
        index = await IndexManager.get_index()
        print("Creating query engine...")
        cls._query_engine = index.as_query_engine()
        print("Query engine created.")
        cls._initialized = True