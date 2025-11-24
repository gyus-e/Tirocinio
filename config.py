__mymodelidx = 0
__test_models = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "google/gemma-3-4b-it",
    "mistralai/Mistral-7B-Instruct-v0.2",
]

documents_dir = "./documents"
model_id = __test_models[__mymodelidx]
max_new_tokens = 512

# RAG configuration settings
use_rag_agent = False
embed_model_id = "nickprock/sentence-bert-base-italian-uncased"
embed_model_path = "./models/embedding-model"

chroma_host = "localhost"
chroma_port = 8000
chroma_collection_name = f"{embed_model_id.split('/')[1]}_embeddings"

chunk_size = 1024
chunk_overlap = 256
retrieve_top_k = 3

temperature = 0.1
generate_top_k = 50
generate_top_p = 0.8
repetition_penalty = 1.1

rag_system_prompt = """
    Hai a disposizione documenti con informazioni relative al catalogo della Biblioteca Pontaniana di Napoli.
    Usa una sola volta la funzione "search_documents" per trovare informazioni utili a rispondere all'utente, poi rispondi utilizzando le informazioni trovate.
    Se non trovi informazioni rilevanti, rispondi con "Non lo so".
"""

# CAG configuration settings
kv_cache_path = f"./kv_cache/kv_cache_{model_id.split('/')[1]}.pt"
cag_system_prompt = """
    Hai a disposizione informazioni relative al catalogo della Biblioteca Pontaniana di Napoli.
"""
cag_answer_instruction = """
    Rispondi utilizzando le informazioni presenti nel contesto fornito.
    Se non trovi informazioni rilevanti, rispondi con "Non lo so".
"""
