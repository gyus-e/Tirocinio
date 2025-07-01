from config import SYSTEM_PROMPT


def build_rag_prompt() -> str:
    rag_prompt = f"""
    {SYSTEM_PROMPT}
    """.strip()

    return rag_prompt
