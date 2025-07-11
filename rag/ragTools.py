from .QueryEngineManager import QueryEngineManager


async def search_documents(query: str) -> str:
    """Useful for answering natural language questions about the provided context."""
    print(f"Using search_documents tool with query: {query}")
    query_engine = await QueryEngineManager.get_query_engine()
    response = await query_engine.aquery(query)
    return str(response)
