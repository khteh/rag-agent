import os, logging, vertexai
from datetime import datetime
from dotenv import load_dotenv
from typing_extensions import List, TypedDict
from google.api_core.exceptions import ResourceExhausted
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain.agents import (
    create_openai_functions_agent,
    Tool,
    AgentExecutor,
)
from langchain import hub
from .HospitalCypherChain import reviews_vector_chain, hospital_cypher_chain
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
from .Tools import TOOLS
load_dotenv()
class RAGAgent():
    _llm = None
    _config = None
    _urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]    
    _prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant named Bob."),
                ("placeholder", "{messages}"),
                ("user", "Remember, always provide accurate answer!"),
        ])
    """
    Prompt
    https://smith.langchain.com/hub
    """
    _rag_prompt = hub.pull("rlm/rag-prompt")
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
        """
        self._llm = init_chat_model("gemini-2.0-flash", model_provider="google_vertexai", streaming=True).bind_tools(TOOLS)
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
    async def CreateGraph(self, config: RunnableConfig) -> CompiledGraph:
        logging.debug(f"\n=== {self.CreateGraph.__name__} ===")
        try:
            # https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent
            return create_react_agent(self._llm, TOOLS, store=InMemoryStore(), checkpointer=MemorySaver(), state_schema=AgentState, name="Healthcare ReAct Agent", prompt=self._rag_prompt)
        except ResourceExhausted as e:
            logging.exception(f"google.api_core.exceptions.ResourceExhausted")

async def make_graph(config: RunnableConfig) -> CompiledGraph:
    return await RAGAgent(config).CreateGraph(config)

async def ChatAgent(agent, config, messages: List[str]):
    logging.info(f"\n=== {ChatAgent.__name__} ===")
    async for event in agent.astream(
        {"messages": [{"role": "user", "content": messages}]},
        stream_mode="values", # Use this to stream all values in the state after each step.
        config=config, # This is needed by Checkpointer
    ):
        event["messages"][-1].pretty_print()

async def main():
    # httpx library is a dependency of LangGraph and is used under the hood to communicate with the AI models.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    config = RunnableConfig(run_name="Healthcare ReAct Agent", thread_id=datetime.now())
    agent = await make_graph(config)
    #show_graph(agent, "RAG ReAct Agent") # This blocks
    """
    graph = agent.get_graph().draw_mermaid_png()
    # Save the PNG data to a file
    with open("/tmp/agent_graph.png", "wb") as f:
        f.write(graph)
    img = Image.open("/tmp/agent_graph.png")
    img.show()
    """
    input_message = ["What is the wait time at Wallace-Hamilton?", "Which hospital has the shortest wait time?"]
    await ChatAgent(agent, config, input_message)

if __name__ == "__main__":
    asyncio.run(main())
