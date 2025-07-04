from llama_index.core import Settings
from llama_index.core.agent.workflow import AgentWorkflow
from .QueryEngineManager import QueryEngineManager
from .rag_prompt import build_rag_prompt
from .ragTools import search_documents


class RAGAgentManager:
    _rag_agent: AgentWorkflow
    _initialized: bool = False

    @classmethod
    async def get_rag_agent(cls) -> AgentWorkflow:
        if not cls._initialized:
            await cls._init_rag_agent()

        return cls._rag_agent

    @classmethod
    async def _init_rag_agent(cls):
        # Initialize query engine (optional, can be done automatically with the first query)
        await QueryEngineManager.get_query_engine()

        # Create an agent workflow
        cls._rag_agent = AgentWorkflow.from_tools_or_functions(
            tools_or_functions=[search_documents],
            llm=Settings.llm,
            system_prompt=build_rag_prompt(),
        )

        cls._initialized = True
