import asyncio, logging, os, vertexai
from uuid_extensions import uuid7, uuid7str
from typing import Annotated, Literal, Sequence
from datetime import datetime
from google.api_core.exceptions import ResourceExhausted
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, MessagesState
from langgraph.graph.graph import (
    END,
    START,
    CompiledGraph,
    Graph,
    Send,
)
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool, ConnectionPool
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate, SystemMessagePromptTemplate
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import List, TypedDict
from langgraph.store.memory import InMemoryStore
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langchain_core.tools import InjectedToolArg, tool
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent, InjectedStore
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel, Field
"""
https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/
https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
https://python.langchain.com/docs/tutorials/qa_chat_history/
https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/message.py
https://langchain-ai.github.io/langgraph/how-tos/streaming/#values
https://python.langchain.com/docs/how_to/configure/
"""
from src.config import config as appconfig
from src.common.State import EmailRAGState, EmailAgentState
from src.utils.image import show_graph
from src.models import ChatMessage
from src.models.EmailModel import EmailModel
from src.models.EscalationModel import EscalationCheckModel
from src.Infrastructure.VectorStore import VectorStore
from src.Infrastructure.Checkpointer import CheckpointerSetup
from data.sample_emails import EMAILS
from src.common.configuration import EmailConfiguration

@tool
async def email_processing_tool(
    email: str, escalation_criteria: str,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> EmailModel:
    """
    Extract structured fields from a regulatory email.
    This should be used when the email message comes from
    a regulatory body or auditor regarding a property or
    construction site that the company works on.

    escalation_criteria is a description of which kinds of
    notices require immediate escalation.

    After calling this tool, you don't need to call any others.
    """
    logging.info(f"\n=== email_processing_tool ===")
    """Extract the user's state from the conversation and update the memory."""
    graph = EmailConfiguration.from_runnable_config(config).graph
    emailState = EmailConfiguration.from_runnable_config(config).email_state
    emailState["email"] = email
    emailState["escalation_text_criteria"] = escalation_criteria
    logging.debug(f"email: {email}, escalation_criteria: {escalation_criteria}, emailState:: {emailState}")
    results = await graph.with_config(config).ainvoke(emailState)
    logging.debug(f"result: {results}")
    return results["extract"]

class EmailRAG():
    _graphName :str = "Email RAG StateGraph"
    _agentName :str = "Email RAG Agent"
    _llm = None
    _chainLLM = None
    _config = None
    _email_parser_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an expert email parser.
                Extract date from the Date: field, name and email from the From: field, project id from the Subject: field or email body text, 
                phone number, site location, violation type, required changes, compliance deadline, and maximum potential fine from the email body text.
                If any of the fields aren't present, don't populate them. Don't populate fields if they're not present in the email.
                Try to cast dates into the dd-mm-YYYY format. Ignore the timestamp and timezone part of the Date. 

                Here's the email:
                {email}
                """,
            ),
            ("human", "{email}"),
            #("placeholder", "{email}"), #should be a list of base messages
        ]
    )
    _escalation_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Determine whether the following email received from a regulatory body requires immediate escalation.
                Immediate escalation is required when {escalation_criteria}.
                Here's the email:
                {email}
                """,
            ),
            ("human", "{email}"),
            #("placeholder", "{email}"), #should be a list of base messages
        ]
    )
    _prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful Email assistant named Bob. Your job is to use the available tools
                                to extract structured fields from an input email and determine if the email needs escalation.
                                escalation_criteria is a description which helps determine if the email needs immediate escalation.
                                After you have received the escalate state from tools, you should stop processing and return an answer with the structured fields extracted from the email."""),
                ("placeholder", "{email}"), #should be a list of base messages
                ("user", "Remember, always provide accurate answer!"),
        ])
    # https://realpython.com/build-llm-rag-chatbot-with-langchain/#chains-and-langchain-expression-language-lcel
    _vectorStore = None
    _in_memory_store: InMemoryStore = None
    _email_parser_chain = None
    _escalation_chain = None
    _graph: CompiledGraph = None
    _agent: CompiledGraph = None
    # Class constructor
    def __init__(self, config: RunnableConfig={}):
        """
        Class EmailRAG Constructor
        """
        logging.info(f"\n=== {self.__init__.__name__} ===")
        self._config = config
        """
        https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
        https://python.langchain.com/docs/integrations/chat/google_vertex_ai_palm/
        https://python.langchain.com/docs/how_to/structured_output/
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
        self._llm = init_chat_model(appconfig.LLM_RAG_MODEL, model_provider="ollama", base_url=appconfig.OLLAMA_URI, configurable_fields=("user_id", "graph", "email_state"), streaming=True, temperature=0).bind_tools([email_processing_tool])
        self._chainLLM = init_chat_model(appconfig.LLM_RAG_MODEL, model_provider="ollama", base_url=appconfig.OLLAMA_URI, temperature=0)
        self._email_parser_chain = (
            self._email_parser_prompt
            | self._chainLLM.with_structured_output(EmailModel)
        )
        self._escalation_chain = (
            self._escalation_prompt
            | self._chainLLM.with_structured_output(EscalationCheckModel)
        )

    async def ParseEmail(self, state: EmailRAGState, config: RunnableConfig) -> EmailRAGState:
        """
        Extract structured fields from a regulatory notice.
        This should be used when the email message comes from
        a regulatory body or auditor regarding a property or
        construction site that the company works on.
        """
        logging.info(f"\n=== {self.ParseEmail.__name__} ===")
        logging.debug(f"state: {state}")
        #print(f"state: {state}, email: {state['email']}")
        state["extract"] = await self._email_parser_chain.with_config(config).ainvoke({"email": state["email"]})
        logging.debug(f"Extract: {state['extract']}")
        return state

    async def NeedsEscalation(self, state: EmailRAGState, config: RunnableConfig) -> EmailRAGState:
        """
        Determine if an email needs escalation
        """
        logging.info(f"\n=== {self.NeedsEscalation.__name__} ===")
        assert self._escalation_chain
        logging.debug(f"state: {state}")
        result: EscalationCheckModel = await self._escalation_chain.with_config(config).ainvoke({"email": state["email"], "escalation_criteria": state["escalation_text_criteria"]})
        logging.debug(f"result: {result}")
        state["escalate"] = (result.needs_escalation or ("max_potential_fine" in state and state["extract"].max_potential_fine and state["extract"].max_potential_fine >= state["escalation_dollar_criteria"]))
        return state

    async def call_agent_model_node(self, state: EmailAgentState, config: RunnableConfig) -> dict[str, list[AIMessage]]:
        """Node to call the email agent model"""
        messages = state["messages"]
        response = await self._llm.with_config(config).ainvoke(messages)
        return {"messages": [response]}

    def route_agent_graph_edge(self, state: EmailAgentState) -> str:
        """Determine whether to call more tools or exit the graph"""
        last_message = state["messages"][-1]
        return "EmailTools" if last_message.tool_calls else END

    async def CreateGraph(self) -> CompiledGraph:
        # Compile application and test
        logging.info(f"\n=== {self.CreateGraph.__name__} ===")
        try:
            graph_builder = StateGraph(EmailRAGState)
            graph_builder.add_node("ParseEmail", self.ParseEmail)
            graph_builder.add_node("NeedsEscalation", self.NeedsEscalation)
            graph_builder.add_edge(START, "ParseEmail")
            graph_builder.add_edge("ParseEmail", "NeedsEscalation")
            graph_builder.add_edge("NeedsEscalation", END)
            self._graph = graph_builder.compile(store=self._in_memory_store, name=self._graphName)
            # https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent
            #self._agent = create_react_agent(self._llm, [email_processing_tool], store=self._in_memory_store, config_schema=EmailConfiguration, state_schema=EmailAgentState, name=self._name, prompt=self._prompt) This doesn't work well as the LLM preprocess the input email instead of passing it through to email_processing_tool as is done in call_agent_model_node
            graph_builder = StateGraph(EmailAgentState)
            graph_builder.add_node("EmailAgent", self.call_agent_model_node)
            graph_builder.add_node("EmailTools", ToolNode([email_processing_tool]))
            graph_builder.add_edge(START, "EmailAgent")
            graph_builder.add_conditional_edges(
                # if the EmailAgent node returns a tool message, your graph moves to the EmailTools node to call the respective tool.
                "EmailAgent", self.route_agent_graph_edge, ["EmailTools", END]
            )
            graph_builder.add_edge("EmailTools", "EmailAgent")
            self._agent = graph_builder.compile(store=self._in_memory_store, name=self._agentName)
        except ResourceExhausted as e:
            logging.exception(f"google.api_core.exceptions.ResourceExhausted")
        return self._agent
    
    def ShowGraph(self):
        show_graph(self._graph, self._graphName) # This blocks
        show_graph(self._agent, self._agentName) # This blocks

    async def Chat(self, criteria, email, email_state, config: RunnableConfig) -> List[str]:
        logging.info(f"\n=== {self.Chat.__name__} ===")
        message_with_criteria = f"The escalation criteria is: {criteria}. Here's the email: {email}"
        result: List[str] = []
        async with AsyncConnectionPool(
            conninfo = appconfig.POSTGRESQL_DATABASE_URI,
            max_size = appconfig.DB_MAX_CONNECTIONS,
            kwargs = appconfig.connection_kwargs,
        ) as pool:
            # Create the AsyncPostgresSaver
            checkpointer = await CheckpointerSetup(pool)
            self._agent.checkpointer = checkpointer
            self._graph.checkpointer = checkpointer
            async for step in self._agent.with_config({"graph": self._graph, "email_state": email_state, "thread_id": uuid7str()}).astream(
                {"messages": [{"role": "user", "content": message_with_criteria}]},
                stream_mode="values",
                #config = config
            ):
                result.append(step["messages"][-1])
                step["messages"][-1].pretty_print()
            return result[-1]

async def make_graph(config: RunnableConfig) -> CompiledGraph:
    return await EmailRAG(config).CreateGraph()

async def main():
    # httpx library is a dependency of LangGraph and is used under the hood to communicate with the AI models.
    #vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    """
    graph = checkpoint_graph.get_graph().draw_mermaid_png()
    # Save the PNG data to a file
    with open("/tmp/checkpoint_graph.png", "wb") as f:
        f.write(graph)
    img = Image.open("/tmp/checkpoint_graph.png")
    img.show()
    """
    config = RunnableConfig(run_name="Email RAG", thread_id=uuid7str(), user_id=uuid7str())
    rag = EmailRAG(config)
    await rag.CreateGraph()
    #rag.ShowGraph()
    email_state = {
        "escalation_dollar_criteria": 100_000,
        "escalation_emails": ["brog@abc.com", "bigceo@company.com"],
    }
    result = await rag.Chat("There's an immediate risk of electrical, water, or fire damage", EMAILS[3], email_state, config)
    assert result
    ai_message = ChatMessage.from_langchain(result)
    assert not ai_message.tool_calls
    assert ai_message.content

if __name__ == "__main__":
    asyncio.run(main())
