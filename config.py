model_id = "meta-llama/Llama-3.2-3B-Instruct"
# model_id = "microsoft/Phi-3-mini-4k-instruct"
# model_id = "google/gemma-3-4b-it"
# model_id = "mistralai/Mistral-7B-Instruct-v0.2"
# model_id = "mistralai/Mistral-7B-Instruct-v0.3"
use_4bit_quantization = True
max_new_tokens = 512

documents_dir = "./documents"

# RAG configuration settings
# embed_model_id = "BAAI/bge-m3"
# embed_model_id = "sentence-transformers/all-MiniLM-L6-v2"
embed_model_id = "nickprock/sentence-bert-base-italian-uncased"
# embed_model_id = "nomic-ai/nomic-embed-text-v1.5"

embed_model_path = "./models/embedding-model"

max_iterations = 5

chroma_host = "localhost"
chroma_port = 8000
chroma_collection_name = f"{embed_model_id.split('/')[1]}_embeddings"

chunk_size = 1024
chunk_overlap = 256
retrieve_top_k = 5

temperature = 0.2
generate_top_k = 40
generate_top_p = 0.8
repetition_penalty = 1.1

rag_system_prompt = """
    Hai a disposizione documenti relativi al catalogo della Biblioteca Pontaniana di Napoli.
    Usa la funzione "search_documents" una sola volta per cercare informazioni utili.
    Dopo aver usato "search_documents", rispondi immediatamente utilizzando le informazioni trovate.
    Se non hai trovato informazioni rilevanti, rispondi con "Non lo so" e termina la risposta.
""".strip()

# CAG configuration settings
kv_cache_path = f"./kv_cache/kv_cache_{model_id.split('/')[1]}.pt"
cag_system_prompt = """
    Rispondi utilizzando le informazioni che ti sono fornite.
    Se non sono sufficienti a rispondere, rispondi con "Non lo so" e termina la risposta.
""".strip()
cag_answer_instructions = """
""".strip()
