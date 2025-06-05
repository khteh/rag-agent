import argparse, atexit, bs4, vertexai, asyncio, logging
from datetime import datetime
from uuid_extensions import uuid7, uuid7str
from typing import Annotated
from typing import Any, Callable, List, Optional, cast
from google.api_core.exceptions import ResourceExhausted
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig, ensure_config
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache
from langgraph.graph.graph import (
    END,
    START,
    Graph,
    Send,
)
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent, InjectedStore
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import List, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langgraph.managed import IsLastStep
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langchain_ollama import OllamaEmbeddings
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain.prompts import PromptTemplate
from psycopg_pool import AsyncConnectionPool, ConnectionPool
from langgraph.store.postgres.aio import AsyncPostgresStore
"""
https://python.langchain.com/docs/tutorials/qa_chat_history/
https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
https://langchain-ai.github.io/langgraph/how-tos/streaming/#values
https://python.langchain.com/docs/how_to/configure/
https://langchain-ai.github.io/langgraph/how-tos/
"""
from src.config import config as appconfig
from src.Infrastructure.VectorStore import VectorStore
from src.common.State import CustomAgentState
from src.utils.image import show_graph
from .Tools import ground_search, store_memory, upsert_memory
from src.Healthcare.Tools import HealthcareReview, HealthcareCypher
from src.Healthcare.HospitalWaitingTime import get_current_wait_times, get_most_available_hospital
from src.common.configuration import Configuration
from src.Infrastructure.PostgreSQLSetup import PostgreSQLCheckpointerSetup, PostgreSQLStoreSetup
from src.Healthcare.prompts import cypher_generation_template, qa_generation_template
class RAGAgent():
    _name:str = "RAG ReAct Agent"
    _llm = None
    _config = None
    _urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        "https://mlflow.org/docs/latest/index.html",
        "https://mlflow.org/docs/latest/tracking/autolog.html",
        "https://mlflow.org/docs/latest/getting-started/tracking-server-overview/index.html",
        "https://mlflow.org/docs/latest/python_api/mlflow.deployments.html",        
    ]
    # https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent
    """
    placeholder:
    Means the template will receive an optional list of messages under the "messages" key.
    A list of the names of the variables for placeholder or MessagePlaceholder that are optional. These variables are auto inferred from the prompt and user need not provide them.
    """
    _prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant named Bob. Always provide accurate answer."),
                ("placeholder", "{messages}")
        ])
    _vectorStore = None
    _tools: List[Callable[..., Any]] = None
    _agent: CompiledStateGraph = None
    #_neo4jGraph: Neo4jGraph = None
    #_cypher_generation_prompt: PromptTemplate = None
    #_qa_generation_prompt: PromptTemplate = None
    #_hospital_cypher_chain: GraphCypherQAChain = None
    # Class constructor
    def __init__(self, config: RunnableConfig={}):
        """
        Class RAGAgent Constructor
        """
        atexit.register(self.Cleanup)
        #vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
        self._config = config
        """
        .bind_tools() gives the agent LLM descriptions of each tool from their docstring and input arguments. 
        If the agent LLM determines that its input requires a tool call, it’ll return a JSON tool message with the name of the tool it wants to use, along with the input arguments.
        For VertexAI, use VertexAIEmbeddings, model="text-embedding-005"; "gemini-2.0-flash" model_provider="google_genai"
        """
        # not a vector store but a LangGraph store object. https://github.com/langchain-ai/langchain/issues/30723
        """
        self._in_memory_store = InMemoryStore(
            index={
                "embed": OllamaEmbeddings(model=appconfig.EMBEDDING_MODEL, base_url=appconfig.OLLAMA_URI, num_ctx=8192, num_gpu=1, temperature=0),
                #"dims": 1536,
            }
        )
        """
        self._vectorStore = VectorStore(model=appconfig.EMBEDDING_MODEL, chunk_size=1000, chunk_overlap=0)
        """
        GoogleSearch ground_search works well but it will sometimes take precedence and overwrite the ingested data into Chhroma and Neo4J. So, exclude it for now until it is really needed.
        """
        self._tools = [self._vectorStore.retriever_tool, HealthcareReview, HealthcareCypher, get_current_wait_times, get_most_available_hospital, upsert_memory]
        self._llm = init_chat_model(appconfig.LLM_RAG_MODEL, model_provider="ollama", base_url = appconfig.OLLAMA_URI, configurable_fields=("user_id"), streaming = True, temperature=0).bind_tools(self._tools)
        """
        self._neo4jGraph = Neo4jGraph(
                url = appconfig.NEO4J_URI,
                username = appconfig.NEO4J_USERNAME,
                password = appconfig.NEO4J_PASSWORD,
            )
        self._cypher_generation_prompt = PromptTemplate(
            input_variables=["schema", "question"], template=cypher_generation_template
        )
        self._qa_generation_prompt = PromptTemplate(
            input_variables=["context", "question"], template=qa_generation_template
        )
        self._hospital_cypher_chain = GraphCypherQAChain.from_llm(
            cypher_llm = self._llm,
            qa_llm = self._llm,
            graph = self._neo4jGraph,
            verbose = True, # Whether intermediate steps your chain performs should be printed.
            qa_prompt = self._qa_generation_prompt,
            cypher_prompt = self._cypher_generation_prompt,
            validate_cypher = True,
            top_k = 100,
            allow_dangerous_requests = True # https://python.langchain.com/docs/security/
        )
        """
        # https://python.langchain.com/docs/integrations/chat/google_vertex_ai_palm/
    
    def Cleanup(self):
        logging.info(f"\n=== {self.Cleanup.__name__} ===")
        #self._in_memory_store.close()

    async def prepare_model_inputs(self, state: CustomAgentState, config: RunnableConfig, store: BaseStore):
        # Retrieve user memories and add them to the system message
        # This function is called **every time** the model is prompted. It converts the state to a prompt
        config = ensure_config(config)
        user_id = config.get("configurable", {}).get("user_id")
        namespace = ("memories", user_id)
        memories = [m.value["data"] for m in await store.asearch(namespace)]
        system_msg = f"User memories: {', '.join(memories)}"
        return [{"role": "system", "content": system_msg}] + state["messages"]

    async def LoadDocuments(self):
        logging.debug(f"\n=== {self.LoadDocuments.__name__} ===")
        await self._vectorStore.LoadDocuments(self._urls)

    async def CreateGraph(self) -> CompiledStateGraph:
        logging.debug(f"\n=== {self.CreateGraph.__name__} ===")
        try:
            """
            https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent
            https://github.com/langchain-ai/langchain/issues/30723
            https://langchain-ai.github.io/langgraph/how-tos/cross-thread-persistence/
            https://github.com/langchain-ai/langgraph/blob/main/libs/prebuilt/langgraph/prebuilt/chat_agent_executor.py#L241
            """
            self._agent = create_react_agent(self._llm, self._tools, config_schema = Configuration, state_schema = CustomAgentState, name = self._name, prompt = self._prompt)
            #self.ShowGraph() # This blocks
        except Exception as e:
            logging.exception(f"Exception! {e}")
        return self._agent

    def ShowGraph(self):
        show_graph(self._agent, self._name) # This blocks

    async def ChatAgent(self, config: RunnableConfig, message: str):
        """
        message is a single string. It can contain multiple questions:
        input_message: str = ("What is task decomposition?\n" "What is the standard method for Task Decomposition?\n" "Once you get the answer, look up common extensions of that method.")
        """
        logging.info(f"\n=== {self.ChatAgent.__name__} ===")
        result: List[str] = []
        async with AsyncConnectionPool(
            conninfo = appconfig.POSTGRESQL_DATABASE_URI,
            max_size = appconfig.DB_MAX_CONNECTIONS,
            kwargs = appconfig.connection_kwargs,
        ) as pool:
            # Create the AsyncPostgresSaver and AsyncPostgresStore
            # https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/state.py#L828
            self._agent.checkpointer = await PostgreSQLCheckpointerSetup(pool)
            self._agent.store = await PostgreSQLStoreSetup(pool)
            """
            https://langchain-ai.github.io/langgraph/concepts/streaming/
            https://langchain-ai.github.io/langgraph/how-tos/#streaming
            """
            #async for step in self._agent.with_config({"user_id": uuid7str()}).astream(
            async for step in self._agent.astream(
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
    """
    https://docs.python.org/3/library/argparse.html
    'store_true' and 'store_false' - These are special cases of 'store_const' used for storing the values True and False respectively. In addition, they create default values of False and True respectively:
    """
    parser = argparse.ArgumentParser(description='Start this LLM-RAG Agent by loading blog content from predefined URLs')
    parser.add_argument('-l', '--load-urls', action='store_true', help='Load documents from URLs')
    args = parser.parse_args()

    # httpx library is a dependency of LangGraph and is used under the hood to communicate with the AI models.
    #vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    config = RunnableConfig(run_name="RAG ReAct Agent", thread_id=uuid7str(), user_id=uuid7str())
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
    print(f"args: {args}")
    if args.load_urls:
        await rag.LoadDocuments()
    input_message: str = ("What is task decomposition?\n" "What is the standard method for Task Decomposition?\n" "Once you get the answer, look up common extensions of that method.")
    #print(f"typeof input_message: {type(input_message)}")
    await rag.ChatAgent(config, input_message)

if __name__ == "__main__":
    asyncio.run(main())
