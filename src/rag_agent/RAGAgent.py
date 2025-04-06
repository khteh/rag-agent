import os, bs4, vertexai, asyncio, logging
from dotenv import load_dotenv
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
from langgraph.graph.graph import (
    END,
    START,
    CompiledGraph,
    Graph,
    Send,
)
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent, InjectedStore
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import List, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langgraph.managed import IsLastStep
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
"""
https://python.langchain.com/docs/tutorials/qa_chat_history/
https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
https://langchain-ai.github.io/langgraph/how-tos/streaming/#values
https://python.langchain.com/docs/how_to/configure/
"""
load_dotenv()
from src.config import config as appconfig
from .VectorStore import VectorStore
from .State import CustomAgentState
from ..utils.image import show_graph
from .Tools import ground_search, save_memory
from .configuration import Configuration
#agent = None
class RAGAgent():
    _name:str = "RAG ReAct Agent"
    _llm = None
    _config = None
    _urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        #"https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        #"https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        #"https://mlflow.org/docs/latest/index.html",
        #"https://mlflow.org/docs/latest/tracking/autolog.html",
        #"https://mlflow.org/docs/latest/getting-started/tracking-server-overview/index.html",
        #"https://mlflow.org/docs/latest/python_api/mlflow.deployments.html",        
    ]    
    _prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant named Bob."),
                ("placeholder", "{messages}"),
                ("human", "Remember, always provide accurate answer!"),
        ])
    _vectorStore = None
    _tools: List[Callable[..., Any]] = None #[vector_store.retriever_tool, ground_search, save_memory]
    _agent: CompiledGraph = None
    # Class constructor
    def __init__(self, config: RunnableConfig={}):
        """
        Class RAGAgent Constructor
        """
        ##vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
        self._config = config
        """
        .bind_tools() gives the agent LLM descriptions of each tool from their docstring and input arguments. 
        If the agent LLM determines that its input requires a tool call, it’ll return a JSON tool message with the name of the tool it wants to use, along with the input arguments.
        For VertexAI, use VertexAIEmbeddings, model="text-embedding-005"; "gemini-2.0-flash" model_provider="google_genai"
        """
        self._vectorStore = VectorStore(model="llama3.3", chunk_size=1000, chunk_overlap=100)
        self._tools = [self._vectorStore.retriever_tool, ground_search, save_memory]
        self._llm = init_chat_model("llama3.3", model_provider="ollama", base_url=appconfig.OLLAMA_URI, streaming=True).bind_tools(self._tools)
        # https://python.langchain.com/docs/integrations/chat/google_vertex_ai_palm/
        """
        self._llm = ChatVertexAI(
                        model="gemini-2.0-flash",
                        temperature=0,
                        max_tokens=None,
                        max_retries=6,
                        stop=None,
                        streaming=True
                    )
        """
    async def prepare_model_inputs(self, state: CustomAgentState, config: RunnableConfig, store: BaseStore):
        # Retrieve user memories and add them to the system message
        # This function is called **every time** the model is prompted. It converts the state to a prompt
        config = ensure_config(config)
        user_id = config.get("configurable", {}).get("user_id")
        namespace = ("memories", user_id)
        memories = [m.value["data"] for m in await store.asearch(namespace)]
        system_msg = f"User memories: {', '.join(memories)}"
        return [{"role": "system", "content": system_msg}] + state["messages"]

    async def CreateGraph(self) -> CompiledGraph:
        logging.debug(f"\n=== {self.CreateGraph.__name__} ===")
        try:
            await self._vectorStore.LoadDocuments(self._urls)
            # https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent
            self._agent = create_react_agent(self._llm, self._tools, store=InMemoryStore(), checkpointer=MemorySaver(), config_schema=Configuration, state_schema=CustomAgentState, name=self._name, prompt=self._prompt)
            #show_graph(self._agent, "RAG ReAct Agent") # This blocks
        except ResourceExhausted as e:
            logging.exception(f"google.api_core.exceptions.ResourceExhausted {e}")
        return self._agent

    def ShowGraph(self):
        show_graph(self._agent, self._name) # This blocks

    async def ChatAgent(self, config: RunnableConfig, messages: List[tuple]): #messages: List[str]):
        logging.info(f"\n=== {self.ChatAgent.__name__} ===")
        async for event in self._agent.with_config({"user_id": uuid7str()}).astream(
            #{"messages": [{"role": "user", "content": messages}]}, This works with gemini-2.0-flash
            {"messages": messages},
            stream_mode="values", # Use this to stream all values in the state after each step.
            config=config, # This is needed by Checkpointer
        ):
            event["messages"][-1].pretty_print()

async def make_graph(config: RunnableConfig) -> CompiledGraph:
    return await RAGAgent(config).CreateGraph()

async def main():
    # httpx library is a dependency of LangGraph and is used under the hood to communicate with the AI models.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    #vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    config = RunnableConfig(run_name="RAG ReAct Agent", thread_id=datetime.now())
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
    input_message = [("human", "What is the standard method for Task Decomposition?"), ("human", "Once you get the answer, look up common extensions of that method.")]
    await rag.ChatAgent(config, input_message)

if __name__ == "__main__":
    asyncio.run(main())
