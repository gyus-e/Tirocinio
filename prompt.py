from DocumentsManager import DocumentsManager
from llama_index.core import Document

def build_context() -> list[Document]:
    documents = DocumentsManager.get_documents()
    return documents


def build_system_prompt():
    documents = build_context()
    context = "\n".join([doc.text for doc in documents])

    system_prompt = f"""
    <|system|>
    You are an assistant who provides concise factual answers based on the context provided.
    <|user|>
    Context: 
    {context}
    Question:
    """.strip()

    return system_prompt
