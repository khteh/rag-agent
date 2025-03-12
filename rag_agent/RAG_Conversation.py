import os, bs4, vertexai,asyncio
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
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
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import List, TypedDict
# https://python.langchain.com/docs/tutorials/qa_chat_history/
load_dotenv()
# https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
vertexai.init(project=os.environ.get("VERTEXAI_PROJECT_ID"), location=os.environ.get("VERTEXAI_PROJECT_LOCATION"))
llm = init_chat_model("gemini-2.0-flash", model_provider="google_vertexai")
embeddings = VertexAIEmbeddings(model="text-embedding-005")
"""
llm = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")"
"""

# https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
vector_store = InMemoryVectorStore(embeddings)

def LoadDocuments(url: str):
    # Load and chunk contents of the blog
    print(f"\n=== {LoadDocuments.__name__} ===")
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    assert len(docs) == 1
    print(f"Total characters: {len(docs[0].page_content)}")
    return docs

def SplitDocuments(docs):
    print(f"\n=== {SplitDocuments.__name__} ===")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    subdocs = text_splitter.split_documents(docs)
    print(f"Split blog post into {len(subdocs)} sub-documents.")
    return subdocs

def IndexChunks(subdocs):
    # Index chunks
    print(f"\n=== {IndexChunks.__name__} ===")
    ids = vector_store.add_documents(documents=subdocs)
    print(f"Document IDs: {ids[:3]}")

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

async def query_or_respond(state: MessagesState, config: RunnableConfig):
    """
    # Step 1: Generate an AIMessage that may include a tool-call to be sent.
    Generate tool call for retrieval or respond
    """
    llm_with_tools = llm.bind_tools([retrieve])
    response = await llm_with_tools.ainvoke(state["messages"], config)
    # MessageState appends messages to state instead of overwriting
    return {"messages": [response]}

# Step 3: Generate a response using the retrieved content.
async def generate(state: MessagesState, config: RunnableConfig):
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
    response = await llm.ainvoke(prompt, config)
    return {"messages": [response]}

def BuildSimpleGraph(config: RunnableConfig) -> StateGraph:
    # Compile application and test
    print(f"\n=== {BuildSimpleGraph.__name__} ===")
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("query_or_respond", query_or_respond)
    graph_builder.add_node("tools", ToolNode([retrieve])) # Execute the retrieval.
    graph_builder.add_node("generate", generate)
    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)
    return graph_builder.compile()

def BuildCheckpointedGraph(config: RunnableConfig) -> StateGraph:
    # Compile application and test
    print(f"\n=== {BuildCheckpointedGraph.__name__} ===")
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("query_or_respond", query_or_respond)
    graph_builder.add_node("tools", ToolNode([retrieve])) # Execute the retrieval.
    graph_builder.add_node("generate", generate)
    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)
    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)

def BuildAgent(config: RunnableConfig) -> StateGraph:
    memory = MemorySaver()
    return create_react_agent(llm, [retrieve], checkpointer=memory)

async def TestDirectResponseWithoutRetrieval(graph, message):
    print(f"\n=== {TestDirectResponseWithoutRetrieval.__name__} ===")
    async for step in graph.astream(
        {"messages": [{"role": "user", "content": message}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()

async def Chat(graph, threadId, messages: List[str]):
    print(f"\n=== {Chat.__name__} ===")
    config = {"configurable": {"thread_id": threadId}}
    for message in messages:
        #input_message = "What is Task Decomposition?"
        async for step in graph.astream(
            {"messages": [{"role": "user", "content": message}]},
            stream_mode="values",
            config = config
        ):
            step["messages"][-1].pretty_print()

async def ChatAgent(agent, threadId, message):
    print(f"\n=== {ChatAgent.__name__} ===")
    config = {"configurable": {"thread_id": threadId}}
    async for event in agent.astream(
        {"messages": [{"role": "user", "content": message}]},
        stream_mode="values",
        config=config,
    ):
        event["messages"][-1].pretty_print()

async def SimpleGraph():
    config = RunnableConfig(run_name="SimpleGraph_RAG_Conversation")
    simple_graph = BuildSimpleGraph(config) # config input parameter is required by langgraph.json to define the graph
    graph = simple_graph.get_graph().draw_mermaid_png()
    # Save the PNG data to a file
    with open("/tmp/simple_graph.png", "wb") as f:
        f.write(graph)
    img = Image.open("/tmp/simple_graph.png")
    img.show()
    await TestDirectResponseWithoutRetrieval(simple_graph, "Hello!")

async def CheckpointedGraph():
    config = RunnableConfig(run_name="CheckpointGraph_RAG_Conversation")
    checkpoint_graph = BuildCheckpointedGraph(config) # config input parameter is required by langgraph.json to define the graph
    graph = checkpoint_graph.get_graph().draw_mermaid_png()
    # Save the PNG data to a file
    with open("/tmp/checkpoint_graph.png", "wb") as f:
        f.write(graph)
    img = Image.open("/tmp/checkpoint_graph.png")
    img.show()
    await Chat(checkpoint_graph, datetime.now(), ["What is Task Decomposition?", "Can you look up some common ways of doing it?"])

async def ReActAgent():
    config = RunnableConfig(run_name="ReAct_RAG_Conversation")
    agent = BuildAgent(config)
    graph = agent.get_graph().draw_mermaid_png()
    # Save the PNG data to a file
    with open("/tmp/agent_graph.png", "wb") as f:
        f.write(graph)
    img = Image.open("/tmp/agent_graph.png")
    img.show()        
    input_message = (
        "What is the standard method for Task Decomposition?\n\n"
        "Once you get the answer, look up common extensions of that method."
    )
    await ChatAgent(agent, datetime.now(), input_message)

async def main():
    #await SimpleGraph()
    #await CheckpointedGraph()
    await ReActAgent()

if __name__ == "__main__":
    docs = LoadDocuments("https://lilianweng.github.io/posts/2023-06-23-agent/")
    subdocs = SplitDocuments(docs)
    IndexChunks(subdocs)
    asyncio.run(main())
