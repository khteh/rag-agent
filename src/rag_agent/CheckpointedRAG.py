import os, bs4, vertexai,asyncio
from State import State
from datetime import datetime
from image import show_graph
from PIL import Image
from typing import Annotated
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
# https://python.langchain.com/docs/tutorials/qa_chat_history/
# https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
"""
llm = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")"
"""

# https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
from Tools import TOOLS
from VectorStore import vector_store

class CheckpointedRAG():
    _llm = None
    _config = None
    _prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant named Bob."),
                ("placeholder", "{messages}"),
                ("user", "Remember, always provide accurate answer!"),
        ])
    # Class constructor
    def __init__(self, config: RunnableConfig={}):
        """
        Class CheckpointedRAG Constructor
        """
        #vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
        self._config = config
        self._llm = init_chat_model("gemini-2.0-flash", model_provider="google_vertexai")
        self._llm = self._llm.bind_tools(TOOLS)

    async def query_or_respond(self, state: MessagesState, config: RunnableConfig):
        """
        # Step 1: Generate an AIMessage that may include a tool-call to be sent.
        Generate tool call for retrieval or respond
        """
        #print(f"state: {state}")
        response = await self._llm.ainvoke(state["messages"], config)
        # MessageState appends messages to state instead of overwriting
        return {"messages": [response]}

    # Step 3: Generate a response using the retrieved content.
    async def generate(self, state: MessagesState, config: RunnableConfig):
        """Generate answer."""
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = f"""
        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer 
        the question. If you don't know the answer, say that you 
        don't know. Use three sentences maximum and keep the 
        answer concise.
        \n\n
        {docs_content}
        """
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # Run
        response = await self._llm.ainvoke(prompt, config)
        return {"messages": [response]}

    async def CreateGraph(self, config: RunnableConfig) -> StateGraph:
        # Compile application and test
        print(f"\n=== {self.CreateGraph.__name__} ===")
        await vector_store.LoadDocuments("https://lilianweng.github.io/posts/2023-06-23-agent/")
        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node("query_or_respond", self.query_or_respond)
        graph_builder.add_node("tools", ToolNode(TOOLS)) # Execute the retrieval.
        graph_builder.add_node("generate", self.generate)
        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)
        return graph_builder.compile(store=InMemoryStore(), checkpointer=MemorySaver(), name="Checkedpoint StateGraph RAG")

async def make_graph(config: RunnableConfig) -> CompiledGraph:
    return await CheckpointedRAG(config).CreateGraph(config)

async def TestDirectResponseWithoutRetrieval(graph, config, message):
    print(f"\n=== {TestDirectResponseWithoutRetrieval.__name__} ===")
    async for step in graph.astream(
        {"messages": [{"role": "user", "content": message}]},
        stream_mode="values",
        config = config
    ):
        step["messages"][-1].pretty_print()

async def Chat(graph, config, messages: List[str]):
    print(f"\n=== {Chat.__name__} ===")
    for message in messages:
        #input_message = "What is Task Decomposition?"
        async for step in graph.astream(
            {"messages": [{"role": "user", "content": message}]},
            stream_mode="values",
            config = config
        ):
            step["messages"][-1].pretty_print()

async def CheckpointedGraph():
    config = RunnableConfig(run_name="Checkedpoint StateGraph RAG", thread_id=datetime.now())
    checkpoint_graph = await make_graph(config) # config input parameter is required by langgraph.json to define the graph
    show_graph(checkpoint_graph, "Checkedpoint StateGraph RAG") # This blocks
    """
    graph = checkpoint_graph.get_graph().draw_mermaid_png()
    # Save the PNG data to a file
    with open("/tmp/checkpoint_graph.png", "wb") as f:
        f.write(graph)
    img = Image.open("/tmp/checkpoint_graph.png")
    img.show()
    """
    await TestDirectResponseWithoutRetrieval(checkpoint_graph, config, "Hello, who are you?")
    await Chat(checkpoint_graph, config, ["What is Task Decomposition?", "Can you look up some common ways of doing it?"])

async def main():
    await CheckpointedGraph()

if __name__ == "__main__":
    asyncio.run(main())
