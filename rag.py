import torch
import logging
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from model import MODEL, TOKENIZER
from storage_context import (
    VECTOR_STORE,
    CHROMA_COLLECTION,
)
from config import (
    MAX_NEW_TOKENS,
    EMBED_MODEL_ID,
    EMBED_MODEL_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVE_TOP_K,
    TEMPERATURE,
    GENERATE_TOP_K,
    GENERATE_TOP_P,
    REPETITION_PENALITY,
    RAG_SYSTEM_PROMPT,
)

Settings.llm = HuggingFaceLLM(
    model=MODEL,
    tokenizer=TOKENIZER,
    device_map="cuda" if torch.cuda.is_available() else "auto",
    max_new_tokens=MAX_NEW_TOKENS,
    generate_kwargs={
        "do_sample": TEMPERATURE > 0.0,
        "temperature": TEMPERATURE if TEMPERATURE > 0.0 else None,
        "top_k": GENERATE_TOP_K if TEMPERATURE > 0.0 else None,
        "top_p": GENERATE_TOP_P if TEMPERATURE > 0.0 else None,
        "repetition_penalty": REPETITION_PENALITY,
    },
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL_ID,
    device="cuda" if torch.cuda.is_available() else None,
    cache_folder=EMBED_MODEL_PATH,
    parallel_process=False,
)
Settings.chunk_size = CHUNK_SIZE
Settings.chunk_overlap = CHUNK_OVERLAP
logging.debug("RAG Settings configured.")

if CHROMA_COLLECTION.count() > 0:
    __INDEX = VectorStoreIndex.from_vector_store(
        vector_store=VECTOR_STORE,
        embed_model=Settings.embed_model,
    )
    logging.debug("RAG Index loaded from existing collection.")
else:
    from documents import DOCUMENTS  # Load the documents only if they are needed

    __INDEX = VectorStoreIndex.from_documents(
        documents=DOCUMENTS,
        storage_context=StorageContext.from_defaults(vector_store=VECTOR_STORE),
        embed_model=Settings.embed_model,
    )
    logging.debug("RAG Index created from documents.")

QUERY_ENGINE = __INDEX.as_query_engine(
    llm=Settings.llm,
    similarity_top_k=RETRIEVE_TOP_K,
)
logging.debug("RAG QueryEngine ready.")


async def search_documents(query: str) -> str:
    """Useful for answering natural language questions about the content of the documents."""
    logging.debug(f'search_documents - Searching documents for query: "{query}".')
    response = await QUERY_ENGINE.aquery(query)
    logging.debug(f'search_documents - Retrieved chunk: "{response}".')
    return str(response)


AGENT = AgentWorkflow.from_tools_or_functions(
    [search_documents],
    llm=Settings.llm,
    system_prompt=RAG_SYSTEM_PROMPT,
)
logging.debug("RAG AgentWorkflow initialized.")

CONTEXT = Context(AGENT)
logging.debug("RAG Context created.")
