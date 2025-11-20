from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from documents import documents
from config import chunk_size, chunk_overlap, embed_model_id, rag_system_prompt
from model import model, tokenizer
from storage_context import (
    storage_context,
    vector_store,
    use_ephemeral_client,
    collection_already_exists,
)


Settings.llm = HuggingFaceLLM(
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)
Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_id)
Settings.chunk_size = chunk_size
Settings.chunk_overlap = chunk_overlap
print("RAG Settings configured.")

index = (
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=Settings.embed_model,
    )
    if use_ephemeral_client or not collection_already_exists
    else VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=Settings.embed_model,
    )
)
print("RAG Index created.")

query_engine = index.as_query_engine()
print("RAG QueryEngine ready.")

async def search_documents(query: str) -> str:
    """Useful for answering natural language questions about the content of the documents."""
    response = await query_engine.aquery(query)
    return str(response)


agent = AgentWorkflow.from_tools_or_functions(
    [search_documents],
    llm=Settings.llm,
    system_prompt=rag_system_prompt,
)
print("RAG AgentWorkflow initialized.")

context = Context(agent)
print("RAG Context created.")
