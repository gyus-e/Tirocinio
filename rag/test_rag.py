from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentWorkflow

from test_questions import questions, delimiter
from utils import ModelConfiguration
from .RAGAgentManager import RAGAgentManager


async def test_rag():
    agent = await RAGAgentManager.get_rag_agent()
    test_context = Context(agent)

    for question in questions:
        print("Q:", question)
        answer = await agent.run(question, ctx=test_context)
        print("RAG:", answer)
        print(delimiter)
