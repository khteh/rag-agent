import asyncio, os, logging, vertexai
from uuid_extensions import uuid7, uuid7str
from src.config import config as appconfig
from datetime import datetime
from typing_extensions import List, TypedDict
from typing import Any, Callable, List, Optional, cast
from google.api_core.exceptions import ResourceExhausted
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain.agents import (
    create_openai_functions_agent,
    Tool,
    AgentExecutor,
)
from langchain import hub
from .HospitalCypherChain import hospital_cypher_chain
from .HospitalReviewChain import reviews_vector_chain
from .HospitalWaitingTime import (
    get_current_wait_times,
    get_most_available_hospital,
)
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langgraph.managed import IsLastStep
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.graph.graph import (
    END,
    START,
    CompiledGraph,
    Graph,
    Send,
)
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent, InjectedStore
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import OllamaEmbeddings
from psycopg_pool import AsyncConnectionPool, ConnectionPool
from .Tools import TOOLS
from src.common.configuration import Configuration
from src.utils.image import show_graph
from src.Infrastructure.VectorStore import VectorStore
from src.Infrastructure.Checkpointer import CheckpointerSetup
class RAGAgent():
    _name:str = "Healthcare ReAct Agent"
    _llm = None
    _config = None
    """
    placeholder:
    Means the template will receive an optional list of messages under the "messages" key.
    A list of the names of the variables for placeholder or MessagePlaceholder that are optional. These variables are auto inferred from the prompt and user need not provide them.
    """
    _prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful healthcare AI assistant named Bob."),
                ("placeholder", "{messages}")
        ])
    _vectorStore = None
    _in_memory_store: InMemoryStore = None
    # Class constructor
    def __init__(self, config: RunnableConfig={}):
        """
        Class RAGAgent Constructor
        """
        #vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
        self._config = config
        """
        .bind_tools() gives the agent LLM descriptions of each tool from their docstring and input arguments. 
        If the agent LLM determines that its input requires a tool call, it’ll return a JSON tool message with the name of the tool it wants to use, along with the input arguments.
        For VertexAI, use VertexAIEmbeddings, model="text-embedding-005"; "gemini-2.0-flash" model_provider="google_genai"
        """
        self._in_memory_store = InMemoryStore(
            index={
                "embed": OllamaEmbeddings(model=appconfig.EMBEDDING_MODEL, base_url=appconfig.OLLAMA_URI, num_ctx=8192, num_gpu=1, temperature=0),
                #"dims": 1536,
            }
        )
        self._vectorStore = VectorStore(model=appconfig.EMBEDDING_MODEL, chunk_size=1000, chunk_overlap=0)
        self._llm = init_chat_model(appconfig.LLM_RAG_MODEL, model_provider="ollama", base_url=appconfig.OLLAMA_URI, streaming=True).bind_tools(TOOLS)
        # https://python.langchain.com/docs/integrations/chat/google_vertex_ai_palm/
    async def CreateGraph(self) -> CompiledGraph:
        logging.debug(f"\n=== {self.CreateGraph.__name__} ===")
        try:
            """
            https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent
            https://github.com/langchain-ai/langchain/issues/30723
            https://langchain-ai.github.io/langgraph/how-tos/cross-thread-persistence/
            """
            self._agent = create_react_agent(self._llm, TOOLS, store = self._in_memory_store, config_schema = Configuration, state_schema=AgentState, name=self._name, prompt=self._prompt)
            #self.ShowGraph() # This blocks
        except ResourceExhausted as e:
            logging.exception(f"google.api_core.exceptions.ResourceExhausted")
        except Exception as e:
            logging.exception(f"Exception! {e}")
        return self._agent

    def ShowGraph(self):
        show_graph(self._agent, self._name) # This blocks

    async def ChatAgent(self, config, messages: List[str]):
        logging.info(f"\n=== {self.ChatAgent.__name__} ===")
        result: List[str] = []
        async with AsyncConnectionPool(
            conninfo = appconfig.POSTGRESQL_DATABASE_URI,
            max_size = appconfig.DB_MAX_CONNECTIONS,
            kwargs = appconfig.connection_kwargs,
        ) as pool:
            # Create the AsyncPostgresSaver
            self._agent.checkpointer = await CheckpointerSetup(pool)
            #async for step in self._agent.with_config({"user_id": uuid7str()}).astream(
            async for step in self._agent.astream(
                #{"messages": [{"role": "user", "content": messages}]}, This works with gemini-2.0-flash
                {"messages": messages}, # This works with Ollama llama3.3
                stream_mode="values", # Use this to stream all values in the state after each step.
                config=config, # This is needed by Checkpointer
            ):
                result.append(step["messages"][-1])
                step["messages"][-1].pretty_print()
            return result[-1]

async def make_graph(config: RunnableConfig) -> CompiledGraph:
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
    input_message = [("human", "Which hospital has the shortest wait time?"), ("human", "What have patients said about their quality of rest during their stay?")]
    await rag.ChatAgent(config, input_message)

if __name__ == "__main__":
    asyncio.run(main())
