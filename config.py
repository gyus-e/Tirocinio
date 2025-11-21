documents_dir = "./documents"
model_id = "meta-llama/Llama-3.2-3B-Instruct"
max_new_tokens = 512

# RAG configuration settings
embed_model_id = "nickprock/sentence-bert-base-italian-uncased"
embed_model_path = "./models/embedding-model"

chroma_host = "localhost"
chroma_port = 8000
chroma_collection_name = "pontaniana-collection"

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
kv_cache_path = "./kv_cache/kv_cache.pt"
cag_system_prompt = """
    Hai a disposizione informazioni relative al catalogo della Biblioteca Pontaniana di Napoli.
"""
cag_answer_instruction = """
    Rispondi utilizzando le informazioni presenti nel contesto che ti è stato fornito.
    Se non trovi informazioni rilevanti, rispondi con "Non lo so".
"""
