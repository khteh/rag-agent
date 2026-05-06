import asyncio, atexit, logging
from datetime import datetime
from uuid_extensions import uuid7, uuid7str
from src.config import config as appconfig
from typing_extensions import List, TypedDict
from typing import Any, Callable, List, Optional, cast
from .HospitalWaitingTime import (
    get_current_wait_times,
    get_most_available_hospital,
)
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.state import CompiledStateGraph
from langchain.agents import create_agent
from psycopg_pool import AsyncConnectionPool
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_ollama import OllamaEmbeddings
from .prompts import HEALTHCARE_INSTRUCTIONS
from .Tools import HealthcareReview, HealthcareCypher, HealthcareMemoryManager, HealthcareMemorySearcher
from src.rag_agent.Tools import upsert_memory, think_tool
from src.Infrastructure.VectorStore import VectorStore
from src.common.Configuration import Configuration
from src.utils.image import show_graph
from src.Infrastructure.PostgreSQLSetup import PostgreSQLCheckpointerSetup, PostgreSQLStoreSetup
from src.common.State import CustomAgentState
class RAGAgent():
    _name:str = "Healthcare Sub-Agent"
    _llm = None
    # placeholder:
    # Means the template will receive an optional list of messages under the "messages" key.
    # A list of the names of the variables for placeholder or MessagePlaceholder that are optional. These variables are auto inferred from the prompt and user need not provide them.
    _closed: bool = False
    _db_pool: AsyncConnectionPool = None
    _self_managed_db_pool: bool = False
    _vectorStore = None
    _checkpointer = None
    _tools: List[Callable[..., Any]] = None
    agent: CompiledStateGraph = None
    def __init__(self, db_pool:AsyncConnectionPool = None, store: VectorStore = None, checkpointer: AsyncPostgresSaver = None):
        """
        Class RAGAgent Constructor

        Args:
           db_pool: Provided by main module app.db_pool if run as hypercorn ASGI application.
           store: Provided by rag_agent.RAGAgent.CreateGraph()
           checkpointer: Provided by rag_agent.RAGAgent.CreateGraph()
        """
        #vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
        # .bind_tools() gives the agent LLM descriptions of each tool from their docstring and input arguments. 
        # If the agent LLM determines that its input requires a tool call, it’ll return a JSON tool message with the name of the tool it wants to use, along with the input arguments.
        # For VertexAI, use VertexAIEmbeddings, model="text-embedding-005"; "gemini-2.0-flash" model_provider="google_genai"
        # not a vector store but a LangGraph store object.
        #atexit.register(self.Cleanup)
        self._self_managed_db_pool = db_pool is None
        self._db_pool = db_pool or AsyncConnectionPool(
                        conninfo = appconfig.POSTGRESQL_DATABASE_URI,
                        max_size = appconfig.DB_MAX_CONNECTIONS,
                        kwargs = appconfig.connection_kwargs,
                        open = False # Opening an async pool in the constructor (using open=True on init) will become an error in a future pool versions. 
                    )
        self._vectorStore = store
        self._checkpointer = checkpointer
        #self._tools = [HealthcareReview, HealthcareCypher, get_current_wait_times, get_most_available_hospital, upsert_memory, think_tool]
        self._tools = [HealthcareReview, HealthcareCypher, get_current_wait_times, get_most_available_hospital, HealthcareMemoryManager, HealthcareMemorySearcher, think_tool]
        if appconfig.OLLAMA_CLOUD_URI:
            self._llm = init_chat_model(appconfig.LLM_RAG_MODEL, model_provider=appconfig.MODEL_PROVIDER, base_url=appconfig.OLLAMA_CLOUD_URI, api_key=appconfig.OLLAMA_API_KEY, streaming=True, temperature=0, think="high")
        else:
            self._llm = init_chat_model(appconfig.LLM_RAG_MODEL, model_provider=appconfig.MODEL_PROVIDER, api_key=appconfig.OLLAMA_API_KEY, streaming=True, temperature=0, think="high")
        # https://python.langchain.com/docs/integrations/chat/google_vertex_ai_palm/
    def __del__(self):
        #self._in_memory_store.close()
        if not self._closed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._Cleanup())
            except Exception as e:
                logging.exception(f"{self.__del__.__name__} exception! {e}")

    async def _Cleanup(self):
        #await self._store.close() AttributeError: 'AsyncPostgresStore' object has no attribute 'close'
        logging.info(f"\n=== {self.__class__.__name__}.{self._Cleanup.__name__} ===")
        if self._self_managed_db_pool:
            await self._db_pool.close()
        self._closed = True

    async def CreateGraph(self) -> CompiledStateGraph:
        logging.info(f"\n=== {self.__class__.__name__}.{self.CreateGraph.__name__} ===")
        try:
            # https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent
            # https://github.com/langchain-ai/langchain/issues/30723
            # https://langchain-ai.github.io/langgraph/how-tos/cross-thread-persistence/
            # https://github.com/langchain-ai/langgraph/blob/main/libs/prebuilt/langgraph/prebuilt/chat_agent_executor.py#L241
            #await self._db_pool.open()
            if self._vectorStore is None:
                self._vectorStore = VectorStore(self._db_pool)
                await self._vectorStore.CreateResources()
            if self._checkpointer is None:
                self._checkpointer = AsyncPostgresSaver(self._db_pool)
                await PostgreSQLCheckpointerSetup(self._db_pool, self._checkpointer)
            self._agent = create_agent(self._llm, self._tools, context_schema = Configuration, state_schema = CustomAgentState, name = self._name, system_prompt = HEALTHCARE_INSTRUCTIONS.format(timestamp=datetime.now()), 
                                       store = self._vectorStore.store, checkpointer = self._checkpointer)
            # self.ShowGraph() # This blocks
        except Exception as e:
            logging.exception(f"Exception! {e}")
        return self._agent
    
    def ShowGraph(self):
        show_graph(self._agent, self._name) # This blocks

    async def ChatAgent(self, config, message: str):
        logging.info(f"\n=== {self.ChatAgent.__name__} ===")
        result: List[str] = []
        async for step in self.agent.astream(
            {"messages": [{"role": "user", "content": message}]},
            stream_mode="values", # Use this to stream all values in the state after each step.
            config=config, # This is needed by Checkpointer
        ):
            result.append(step["messages"][-1])
            step["messages"][-1].pretty_print()
        return result[-1]

async def make_graph() -> CompiledStateGraph:
    return await RAGAgent().CreateGraph()

async def main():
    # httpx library is a dependency of LangGraph and is used under the hood to communicate with the AI models.
    #vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    config = RunnableConfig(run_name="Healthcare Sub-Agent", configurable={"thread_id": uuid7str(), "user_id": uuid7str()})
    rag = RAGAgent()
    await rag.CreateGraph()
    """
    graph = agent.get_graph().draw_mermaid_png()
    # Save the PNG data to a file
    with open("/tmp/agent_graph.png", "wb") as f:
        f.write(graph)
    img = Image.open("/tmp/agent_graph.png")
    img.show()
    """
    await rag.ChatAgent(config, "Which hospital has the shortest wait time?")
    await rag.ChatAgent(config, "What have patients said about their quality of rest during their stay?")

if __name__ == "__main__":
    asyncio.run(main())
