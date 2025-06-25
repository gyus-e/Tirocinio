from llama_index.core import Document
from params import SYSTEM_PROMPT
from ..DocumentsManager import DocumentsManager

def build_context() -> list[Document]:
    documents = DocumentsManager.get_documents()
    return documents


def build_cag_prompt():
    documents = build_context()
    context = "\n".join([doc.text for doc in documents])

    system_prompt = f"""
    <|system|>
    {SYSTEM_PROMPT}
    <|user|>
    Context: 
    {context}
    Question:
    """.strip()

    return system_prompt
