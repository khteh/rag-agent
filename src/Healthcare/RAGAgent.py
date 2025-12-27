import asyncio, logging
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
from psycopg_pool import AsyncConnectionPool, ConnectionPool
from langgraph.store.postgres.aio import AsyncPostgresStore
from .prompts import HEALTHCARE_INSTRUCTIONS
from .Tools import HealthcareReview, HealthcareCypher
from src.rag_agent.Tools import upsert_memory, think_tool
from src.common.Configuration import Configuration
from src.utils.image import show_graph
from src.Infrastructure.VectorStore import VectorStore
from src.Infrastructure.PostgreSQLSetup import PostgreSQLCheckpointerSetup, PostgreSQLStoreSetup
from src.common.State import CustomAgentState
class RAGAgent():
    _name:str = "Healthcare ReAct Agent"
    _llm = None
    _config = None
    # placeholder:
    # Means the template will receive an optional list of messages under the "messages" key.
    # A list of the names of the variables for placeholder or MessagePlaceholder that are optional. These variables are auto inferred from the prompt and user need not provide them.
    _prompt = "You are a helpful healthcare AI assistant named Bob. Always provide accurate answer."
    _db_pool: AsyncConnectionPool = None
    _store: AsyncPostgresStore = None   
    _tools: List[Callable[..., Any]] = None
    agent: CompiledStateGraph = None
    # Class constructor
    def __init__(self, config: RunnableConfig={}):
        """
        Class RAGAgent Constructor
        """
        #vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
        self._config = config
        # .bind_tools() gives the agent LLM descriptions of each tool from their docstring and input arguments. 
        # If the agent LLM determines that its input requires a tool call, it’ll return a JSON tool message with the name of the tool it wants to use, along with the input arguments.
        # For VertexAI, use VertexAIEmbeddings, model="text-embedding-005"; "gemini-2.0-flash" model_provider="google_genai"
        # not a vector store but a LangGraph store object.
        #self._in_memory_store = InMemoryStore(
        #    index={
        #        "embed": OllamaEmbeddings(model=appconfig.EMBEDDING_MODEL, base_url=appconfig.BASE_URI, num_ctx=8192, num_gpu=1, temperature=0),
        #        #"dims": 1536,
        #    }
        #)
        self._db_pool = AsyncConnectionPool(
                conninfo = appconfig.POSTGRESQL_DATABASE_URI,
                max_size = appconfig.DB_MAX_CONNECTIONS,
                kwargs = appconfig.connection_kwargs,
            )
        self._tools = [HealthcareReview, HealthcareCypher, get_current_wait_times, get_most_available_hospital, upsert_memory, think_tool]
        #self._llm = init_chat_model(appconfig.LLM_RAG_MODEL, model_provider="ollama", base_url=appconfig.BASE_URI, streaming=True, temperature=0).bind_tools(self._tools)
        if appconfig.BASE_URI:
            self._llm = init_chat_model(appconfig.LLM_RAG_MODEL, model_provider=appconfig.MODEL_PROVIDER, base_url=appconfig.BASE_URI, streaming=True, temperature=0)
        else:
            self._llm = init_chat_model(appconfig.LLM_RAG_MODEL, model_provider=appconfig.MODEL_PROVIDER, streaming=True, temperature=0)
        # https://python.langchain.com/docs/integrations/chat/google_vertex_ai_palm/

    async def Cleanup(self):
        logging.info(f"\n=== {self.Cleanup.__name__} ===")
        await self._store.close()
        await self._db_pool.close()
        #self._in_memory_store.close()

    async def CreateGraph(self) -> CompiledStateGraph:
        logging.debug(f"\n=== {self.CreateGraph.__name__} ===")
        try:
            # https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent
            # https://github.com/langchain-ai/langchain/issues/30723
            # https://langchain-ai.github.io/langgraph/how-tos/cross-thread-persistence/
            # https://github.com/langchain-ai/langgraph/blob/main/libs/prebuilt/langgraph/prebuilt/chat_agent_executor.py#L241
            await self._db_pool.open()
            self._store = await PostgreSQLStoreSetup(self._db_pool) # store is needed when creating the ReAct agent / StateGraph for InjectedStore to work
            self._agent = create_agent(self._llm, self._tools, context_schema = Configuration, state_schema=CustomAgentState, name=self._name, system_prompt=HEALTHCARE_INSTRUCTIONS, store = self._store)
            #self.ShowGraph() # This blocks
        except Exception as e:
            logging.exception(f"Exception! {e}")
        return self.agent

    def ShowGraph(self):
        show_graph(self.agent, self._name) # This blocks

    async def ChatAgent(self, config, message: str):
        logging.info(f"\n=== {self.ChatAgent.__name__} ===")
        result: List[str] = []
        async with AsyncConnectionPool(
            conninfo = appconfig.POSTGRESQL_DATABASE_URI,
            max_size = appconfig.DB_MAX_CONNECTIONS,
            kwargs = appconfig.connection_kwargs,
        ) as pool:
            # Create the AsyncPostgresSaver
            self.agent.checkpointer = await PostgreSQLCheckpointerSetup(pool)
            async for step in self.agent.astream(
                {"messages": [{"role": "user", "content": message}]},
                stream_mode="values", # Use this to stream all values in the state after each step.
                config=config, # This is needed by Checkpointer
            ):
                result.append(step["messages"][-1])
                step["messages"][-1].pretty_print()
            return result[-1]

async def make_graph(config: RunnableConfig) -> CompiledStateGraph:
    return await RAGAgent(config).CreateGraph()

async def main():
    # httpx library is a dependency of LangGraph and is used under the hood to communicate with the AI models.
    #vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    config = RunnableConfig(run_name="Healthcare ReAct Agent", thread_id=uuid7str(), user_id=uuid7str())
    rag = RAGAgent(config)
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
