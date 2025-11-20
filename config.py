documents_dir = "./documents"
model_id = "meta-llama/Llama-3.2-3B-Instruct"

# RAG configuration settings
embed_model_id = "BAAI/bge-m3"
chroma_host = "localhost"
chroma_port = 8000
chroma_collection_name = "pontaniana-collection"
chunk_size = 1024
chunk_overlap = 128
rag_system_prompt = """
    Sei un assistente bibliotecario per la Biblioteca Pontaniana di Napoli.
    Hai a disposizione dei documenti con informazioni relative al catalogo della biblioteca.
    Cerca nei documenti forniti le informazioni utili per rispondere all'utente, poi rispondi utilizzando le informazioni trovate.
    Se non trovi informazioni rilevanti, rispondi con "Non lo so".
"""

# CAG configuration settings
kv_cache_path = "./kv_cache/kv_cache.pt"
cag_system_prompt = """
    Sei un assistente bibliotecario per la Biblioteca Pontaniana di Napoli.
    Hai a disposizione informazioni relative al catalogo della biblioteca.
    Rispondi utilizzando le informazioni presenti nel contesto fornito.
    Se non trovi informazioni rilevanti, rispondi con "Non lo so".
"""
cag_answer_instruction = """
"""
