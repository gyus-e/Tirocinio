from .QueryEngineManager import QueryEngineManager

async def search_documents(query: str) -> str:
    """Executes a natural language query to retrieve relevant information from the indexed documents."""
    print(f"Using search_documents tool with query: {query}")
    query_engine = await QueryEngineManager.get_query_engine()
    response = await query_engine.aquery(query)
    print("search_documents response:", response)
    return str(response)
