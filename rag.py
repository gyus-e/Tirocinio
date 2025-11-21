import torch
import logging
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from model import model, tokenizer
from storage_context import (
    vector_store,
    chroma_collection,
)
from config import (
    max_new_tokens,
    embed_model_id,
    embed_model_path,
    chunk_size,
    chunk_overlap,
    retrieve_top_k,
    temperature,
    generate_top_k,
    generate_top_p,
    repetition_penalty,
    rag_system_prompt,
)

Settings.llm = HuggingFaceLLM(
    model=model,
    tokenizer=tokenizer,
    device_map="cuda" if torch.cuda.is_available() else "auto",
    max_new_tokens=max_new_tokens,
    generate_kwargs={
        "do_sample": temperature > 0.0,
        "temperature": temperature if temperature > 0.0 else None,
        "top_k": generate_top_k if temperature > 0.0 else None,
        "top_p": generate_top_p if temperature > 0.0 else None,
        "repetition_penalty": repetition_penalty,
    },
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name=embed_model_id,
    device="cuda" if torch.cuda.is_available() else None,
    cache_folder=embed_model_path,
    parallel_process=False,
)
Settings.chunk_size = chunk_size
Settings.chunk_overlap = chunk_overlap
logging.debug("RAG Settings configured.")

if chroma_collection.count() > 0:
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=Settings.embed_model,
    )
    logging.debug("RAG Index loaded from existing collection.")
else:
    from documents import documents  # Load the documents only if they are needed

    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
        embed_model=Settings.embed_model,
    )
    logging.debug("RAG Index created from documents.")

query_engine = index.as_query_engine(
    llm=Settings.llm,
    similarity_top_k=retrieve_top_k,
)
logging.debug("RAG QueryEngine ready.")


async def search_documents(query: str) -> str:
    """Useful for answering natural language questions about the content of the documents."""
    logging.debug(f'search_documents - Searching documents for query: "{query}".')
    response = await query_engine.aquery(query)
    logging.debug(f'search_documents - Retrieved chunk: "{response}".')
    return str(response)


agent = AgentWorkflow.from_tools_or_functions(
    [search_documents],
    llm=Settings.llm,
    system_prompt=rag_system_prompt,
)
logging.debug("RAG AgentWorkflow initialized.")

context = Context(agent)
logging.debug("RAG Context created.")
