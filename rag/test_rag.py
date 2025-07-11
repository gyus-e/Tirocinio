from llama_index.core.workflow import Context

from test_questions import questions, delimiter
from .RAGAgentManager import RAGAgentManager
from . import initialize_settings

async def test_rag():
    initialize_settings()
    agent = await RAGAgentManager.get_rag_agent()
    test_context = Context(agent)

    for question in questions:
        print("Q:", question)
        answer = await agent.run(question, ctx=test_context)
        print("RAG:", answer)
        print(delimiter)
