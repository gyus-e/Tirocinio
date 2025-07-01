from llama_index.core.workflow import Context

from test_questions import questions, delimiter
from .RAGAgentManager import RAGAgentManager


async def run_rag():
    agent = await RAGAgentManager.get_rag_agent()
    test_context = Context(agent)

    print("Press Ctrl+C to exit.")
    while True:
        question = input()
        answer = await agent.run(question, ctx=test_context)
        print("RAG:", answer)
        print(delimiter)
