from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from documents import documents
from model import model, tokenizer, embed_model_id
from storage_context import (
    storage_context,
    vector_store,
    ephemeral_client,
    collection_exists,
)


Settings.llm = HuggingFaceLLM(
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)
Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_id)
Settings.chunk_size = 1024
Settings.chunk_overlap = 128


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

context = Context(agent)
