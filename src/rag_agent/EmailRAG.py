import asyncio, logging, os, vertexai
from dotenv import load_dotenv
from typing import Annotated, Literal, Sequence
from datetime import datetime
from google.api_core.exceptions import ResourceExhausted
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, MessagesState
from langgraph.graph.graph import (
    END,
    START,
    CompiledGraph,
    Graph,
    Send,
)
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import List, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langgraph.store.memory import InMemoryStore
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langchain_core.tools import InjectedToolArg, tool
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent, InjectedStore
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
load_dotenv()
from .State import EmailRAGState, EmailAgentState
from src.utils.image import show_graph
from src.schema.EmailModel import EmailModel
from src.schema.EscalationModel import EscalationCheckModel
from .VectorStore import VectorStore
from data.sample_emails import EMAILS
from .configuration import EmailConfiguration
@tool
async def email_processing_tool(
    email: str, escalation_criteria: str,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> EmailModel:
    """
    Extract structured fields from a regulatory notice.
    This should be used when the email message comes from
    a regulatory body or auditor regarding a property or
    construction site that the company works on.

    escalation_criteria is a description of which kinds of
    notices require immediate escalation.

    After calling this tool, you don't need to call any others.
    """
    logging.info(f"\n=== {email_processing_tool.__init__.__name__} ===")
    """Extract the user's state from the conversation and update the memory."""
    graph = EmailConfiguration.from_runnable_config(config).graph
    emailState = EmailConfiguration.from_runnable_config(config).email_state
    emailState["notice_message"] = email
    emailState["escalation_text_criteria"] = escalation_criteria
    #print(f"emailState: {emailState}")
    results = await graph.ainvoke(emailState)
    return results["notice_email_extract"]

class EmailRAG():
    _name :str = "Email RAG Agent"
    _llm = None
    _config = None
    _email_parser_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Parse the date of notice, sending entity name, sending entity
                phone, sending entity email, project id, site location,
                violation type, required changes, compliance deadline, and
                maximum potential fine from the message. If any of the fields
                aren't present, don't populate them. Try to cast dates into
                the YYYY-mm-dd format. Don't populate fields if they're not
                present in the message.

                Here's the notice message:

                {message}
                """,
            ),
            #("placeholder", "{message}"),
        ]
    )
    _escalation_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Determine whether the following notice received
                from a regulatory body requires immediate escalation.
                Immediate escalation is required when {escalation_criteria}.

                Here's the notice message:

                {message}
                """,
            ),
            #("placeholder", "{message}"),
        ]
    )
    _prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful Email assistant named Bob."),
                ("placeholder", "{messages}"),
                ("user", "Remember, always provide accurate answer!"),
        ])
    # https://realpython.com/build-llm-rag-chatbot-with-langchain/#chains-and-langchain-expression-language-lcel
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
        # https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
        self._llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai", configurable_fields=("user_id", "graph", "email_state"), streaming=True)
        # https://python.langchain.com/docs/integrations/chat/google_vertex_ai_palm/
        # https://python.langchain.com/docs/how_to/structured_output/
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
        self._llm = self._llm.bind_tools([email_processing_tool])
        self._email_parser_chain = (
            self._email_parser_prompt
            | self._llm.with_structured_output(EmailModel)
        )
        self._escalation_chain = (
            self._escalation_prompt
            | self._llm.with_structured_output(EscalationCheckModel)
        )

    async def ParseEmail(self, config: RunnableConfig, *, state: EmailRAGState) -> EmailRAGState:
        """
        Use the EmailModel LCEL to extract fields from email
        """
        logging.info(f"\n=== {self.ParseEmail.__name__} ===")
        state["notice_email_extract"] = await self._email_parser_chain.ainvoke({"message": state["notice_message"]}, config) if state["notice_message"] else None
        #state["notice_email_extract"] = await self._email_parser_chain.ainvoke({"message": [{"role": "user", "contents": state["notice_message"]}]}, self._config) if state["notice_message"] else None
        return state

    async def NeedsEscalation(self, config: RunnableConfig, *, state: EmailRAGState) -> EmailRAGState:
        """
        Determine if an email needs escalation
        """
        logging.info(f"\n=== {self.NeedsEscalation.__name__} ===")
        result: EscalationCheckModel = await self._escalation_chain.ainvoke({"message": state["notice_message"], "escalation_criteria": state["escalation_text_criteria"]}, config) if state and state["notice_message"] else None
        state["requires_escalation"] = (result.needs_escalation or state["notice_email_extract"].max_potential_fine >= state["escalation_dollar_criteria"])
        return state

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
            self._graph = graph_builder.compile(store=InMemoryStore(), checkpointer=MemorySaver(), name=self._name)
            # https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent
            self._agent = create_react_agent(self._llm, [email_processing_tool], store=InMemoryStore(), checkpointer=MemorySaver(), config_schema=EmailConfiguration, state_schema=EmailAgentState, name=self._name, prompt=self._prompt)
        except ResourceExhausted as e:
            logging.exception(f"google.api_core.exceptions.ResourceExhausted")
        return self._agent
    
    def ShowGraph(self):
        show_graph(self._agent, self._name) # This blocks

    async def Chat(self, config: RunnableConfig):
        logging.info(f"\n=== {self.Chat.__name__} ===")
        escalation_criteria = """"There's an immediate risk of electrical, water, or fire damage"""
        message_with_criteria = f"""The escalation criteria is: {escalation_criteria}
                                    Here's the email:
                                    {EMAILS[3]}
                                    """
        email_state = {
            "escalation_dollar_criteria": 100_000,
            "escalation_emails": ["brog@abc.com", "bigceo@company.com"],
        }
        async for step in self._agent.astream(
            {"messages": [{"role": "user", "content": message_with_criteria}]},
            {"configurable": {"graph": self._graph, "email_state": email_state}},
            stream_mode="values",
            config = config
        ):
            step["messages"][-1].pretty_print()

async def make_graph(config: RunnableConfig) -> CompiledGraph:
    return await EmailRAG(config).CreateGraph()

async def main():
    # httpx library is a dependency of LangGraph and is used under the hood to communicate with the AI models.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    """
    graph = checkpoint_graph.get_graph().draw_mermaid_png()
    # Save the PNG data to a file
    with open("/tmp/checkpoint_graph.png", "wb") as f:
        f.write(graph)
    img = Image.open("/tmp/checkpoint_graph.png")
    img.show()
    """
    config = RunnableConfig(run_name="Email RAG Test", thread_id=datetime.now())
    rag = EmailRAG(config)
    await rag.CreateGraph()
    rag.ShowGraph()
    await rag.Chat(config)

if __name__ == "__main__":
    asyncio.run(main())
