import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

from cag_utils import (
    get_kv_cache,
    get_kv_len,
    generate_response,
    clean_up,
)
from documents import DOCUMENTS
from model import MODEL, TOKENIZER, EMBED_DEVICE
from config import (
    CAG_SYSTEM_PROMPT,
    KV_CACHE_PATH,
    MAX_NEW_TOKENS,
)


torch.serialization.add_safe_globals([DynamicCache, DynamicLayer])

__DOC_TEXT = "\n".join(
    [f"{doc.metadata['file_name']}\n{doc.text}" for doc in DOCUMENTS]
).strip()

__KV = get_kv_cache(
    KV_CACHE_PATH,
    CAG_SYSTEM_PROMPT,
    __DOC_TEXT,
    MODEL,
    TOKENIZER,
    EMBED_DEVICE,
)

__KV_LEN = get_kv_len(__KV)


def cag_query(query: str) -> str:
    """Useful for answering natural language questions using the cached knowledge."""
    response = generate_response(
        __KV,
        query,
        MODEL,
        TOKENIZER,
        EMBED_DEVICE,
        MAX_NEW_TOKENS,
    )
    clean_up(__KV, __KV_LEN)
    return str(response)
