import asyncio, logging, os, vertexai
from dotenv import load_dotenv
from typing import Annotated, Literal, Sequence
from datetime import datetime
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
from pydantic import BaseModel, Field
"""
https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/
https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
https://python.langchain.com/docs/tutorials/qa_chat_history/
https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/message.py
https://langchain-ai.github.io/langgraph/how-tos/streaming/#values
"""
load_dotenv()
from .State import EmailRAGState
from src.utils.image import show_graph
from src.schema.EmailModel import EmailModel
from src.schema.EscalationModel import EscalationCheckModel
from .VectorStore import vector_store

class EmailRAG():
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
    # https://realpython.com/build-llm-rag-chatbot-with-langchain/#chains-and-langchain-expression-language-lcel
    _email_parser_chain = None
    _escalation_chain = None
    # Class constructor
    def __init__(self, config: RunnableConfig={}):
        """
        Class EmailRAG Constructor
        """
        logging.info(f"\n=== {self.__init__.__name__} ===")
        self._config = config
        # https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
        self._llm = init_chat_model("gemini-2.0-flash", model_provider="google_vertexai", streaming=True)
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
        self._llm = self._llm.bind_tools([vector_store.retriever_tool])
        self._email_parser_chain = (
            self._email_parser_prompt
            | self._llm.with_structured_output(EmailModel)
        )
        self._escalation_chain = (
            self._escalation_prompt
            | self._llm.with_structured_output(EscalationCheckModel)
        )

    async def ParseEmail(self, state: EmailRAGState) -> EmailRAGState:
        """
        Use the EmailModel LCEL to extract fields from email
        """
        logging.info(f"\n=== {self.ParseEmail.__name__} ===")
        #{"messages": [{"role": "user", "content": message}]}
        state["notice_email_extract"] = await self._email_parser_chain.ainvoke({"message": state["notice_message"]}, self._config) if state["notice_message"] else None
        return state

    async def NeedsEscalation(self, state: EmailRAGState) -> EmailRAGState:
        """
        Determine if an email needs escalation
        """
        logging.info(f"\n=== {self.NeedsEscalation.__name__} ===")
        result: EscalationCheckModel = await self._escalation_chain.ainvoke({"message": state["notice_message"], "escalation_criteria": state["escalation_text_criteria"]}, self._config) if state and state["notice_message"] else None
        state["requires_escalation"] = (result.needs_escalation or state["notice_email_extract"].max_potential_fine >= state["escalation_dollar_criteria"])
        return state

    async def CreateGraph(self, config: RunnableConfig) -> StateGraph:
        # Compile application and test
        logging.info(f"\n=== {self.CreateGraph.__name__} ===")
        graph_builder = StateGraph(EmailRAGState)
        graph_builder.add_node("ParseEmail", self.ParseEmail)
        graph_builder.add_node("NeedsEscalation", self.NeedsEscalation)
        graph_builder.add_edge(START, "ParseEmail")
        graph_builder.add_edge("ParseEmail", "NeedsEscalation")
        graph_builder.add_edge("NeedsEscalation", END)
        return graph_builder.compile(store=InMemoryStore(), checkpointer=MemorySaver(), name="Email RAG")

async def make_graph(config: RunnableConfig) -> CompiledGraph:
    return await EmailRAG(config).CreateGraph(config)

async def Chat(graph, config, messages: List[str]):
    logging.info(f"\n=== {Chat.__name__} ===")
    for message in messages:
        #input_message = "What is Task Decomposition?"
        async for step in graph.astream(
            {"messages": [{"role": "user", "content": message}]},
            stream_mode="values",
            config = config
        ):
            step["messages"][-1].pretty_print()

async def main():
    # httpx library is a dependency of LangGraph and is used under the hood to communicate with the AI models.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    config = RunnableConfig(run_name="Email RAG", thread_id=datetime.now())
    emailrag_graph = await make_graph(config) # config input parameter is required by langgraph.json to define the graph
    show_graph(emailrag_graph, "Email RAG") # This blocks
    """
    graph = checkpoint_graph.get_graph().draw_mermaid_png()
    # Save the PNG data to a file
    with open("/tmp/checkpoint_graph.png", "wb") as f:
        f.write(graph)
    img = Image.open("/tmp/checkpoint_graph.png")
    img.show()
    """
    #await Chat(emailrag_graph, config, ["What is Task Decomposition?", "Can you look up some common ways of doing it?"])

if __name__ == "__main__":
    asyncio.run(main())
