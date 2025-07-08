from llama_index.core import Settings
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentWorkflow
from .QueryEngineManager import QueryEngineManager
from .rag_prompt import build_rag_prompt
from .ragTools import search_documents


class RAGAgentManager:
    _rag_agent: AgentWorkflow
    _context: Context
    __initialized: bool = False
    __has_context: bool = False

    @classmethod
    async def get_rag_agent(cls) -> AgentWorkflow:
        if not cls.__initialized:
            await cls._init_rag_agent()

        return cls._rag_agent

    @classmethod
    async def get_context(cls) -> Context:
        if not cls.__initialized:
            await cls._init_rag_agent()

        if not cls.__has_context:
            cls._context = Context(cls._rag_agent)
            cls.__has_context = True

        return cls._context

    @classmethod
    async def reset_context(cls):
        if not cls.__initialized or not cls.__has_context:
            return

        cls._context = Context(cls._rag_agent)
        return cls._context

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

        cls.__initialized = True
