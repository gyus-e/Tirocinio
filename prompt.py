def build_context():
    with open("documents/document.txt", "r", encoding="utf-8") as file:
        doc_text = file.read()
    return doc_text


def build_system_prompt():
    context = build_context()

    system_prompt = f"""
    <|system|>
    You are an assistant who provides concise factual answers based on the context provided.
    <|user|>
    Context: 
    {context}
    Question:
    """.strip()

    return system_prompt
