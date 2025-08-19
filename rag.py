from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from knowledge import documents


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

index = (
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=Settings.embed_model,
    )
    if ephemeral_client or not collection_exists
    else VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=Settings.embed_model,
    )
)

query_engine = index.as_query_engine()


async def search_documents(query: str) -> str:
    """Useful for answering natural language questions about the content of the documents."""
    response = await query_engine.aquery(query)
    return str(response)


agent = AgentWorkflow.from_tools_or_functions(
    [search_documents],
    llm=Settings.llm,
    system_prompt="""You are a helpful assistant that can search through documents to answer questions.""",
)
