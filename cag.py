import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

from cag_utils import (
    load_kv_cache,
    create_kv_cache,
    get_kv_len,
    generate_response,
    clean_up,
)
from model import MODEL, TOKENIZER, EMBED_DEVICE
from config import (
    CAG_SYSTEM_PROMPT,
    KV_CACHE_PATH,
    MAX_NEW_TOKENS,
)


torch.serialization.add_safe_globals([DynamicCache, DynamicLayer])


try:
    __KV = load_kv_cache(KV_CACHE_PATH)

except FileNotFoundError:
    from documents import DOCUMENTS

    __DOC_TEXT = "\n".join(
        [
            f"""
            {doc.metadata['file_name']}
            {doc.text.strip()}
            """.strip()
            for doc in DOCUMENTS
        ]
    ).strip()

    __KV = create_kv_cache(
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
        f"{query}\nRisposta:\n",  # Evita che il modello generi altre domande oppure la scritta "Risposta:"
        MODEL,
        TOKENIZER,
        EMBED_DEVICE,
        MAX_NEW_TOKENS,
    )
    clean_up(__KV, __KV_LEN)
    return response
