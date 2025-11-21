import bs4, asyncio, logging
from src.utils.image import show_graph
from src.common.State import State
from langchain.chat_models import init_chat_model
from langchain_classic import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import (
    START,
    StateGraph,
)
from src.config import config as appconfig
from .Tools import upsert_memory
# https://python.langchain.com/docs/tutorials/rag/
"""
https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
#vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
For VertexAI, use VertexAIEmbeddings, model="text-embedding-005"; "gemini-2.0-flash" model_provider="google_genai"
https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
embeddings = VertexAIEmbeddings(model="text-embedding-005")
"""
_vectorStore = VectorStore(model=appconfig.EMBEDDING_MODEL, chunk_size=1000, chunk_overlap=0)
_tools = [_vectorStore.retriever_tool, upsert_memory]
llm = init_chat_model(appconfig.LLM_RAG_MODEL, model_provider="ollama", base_url=appconfig.OLLAMA_URI, streaming=True, temperature=0).bind_tools(_tools)
"""
llm = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")"
"""
from src.Infrastructure.VectorStore import VectorStore
vector_store = VectorStore(model=appconfig.EMBEDDING_MODEL, chunk_size=1000, chunk_overlap=0)

def LoadDocuments(url: str):
    # Load and chunk contents of the blog
    logging.info(f"\n=== {LoadDocuments.__name__} ===")
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
    logging.debug(f"Total characters: {len(docs[0].page_content)}")
    return docs

def SplitDocuments(docs):
    logging.info(f"\n=== {SplitDocuments.__name__} ===")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    subdocs = text_splitter.split_documents(docs)
    logging.debug(f"Split blog post into {len(subdocs)} sub-documents.")
    return subdocs

async def IndexChunks(subdocs):
    # Index chunks
    logging.info(f"\n=== {IndexChunks.__name__} ===")
    ids = await vector_store.aadd_documents(documents=subdocs)
    logging.debug(f"Document IDs: {ids[:3]}")

# Define application steps
async def retrieve(state: State):
    retrieved_docs = await vector_store.asimilarity_search(state["question"])
    return {"context": retrieved_docs}

async def generate(state: State, config: RunnableConfig):
    """
    Define prompt for question-answering
    https://smith.langchain.com/hub
    """
    prompt = hub.pull("rlm/rag-prompt")
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = await prompt.ainvoke({"question": state["question"], "context": docs_content}, config)
    response = await llm.ainvoke(messages, config)
    return {"answer": response.content}

def BuildGraph(config: RunnableConfig) -> CompiledStateGraph:
    # Compile application and test
    logging.info(f"\n=== {BuildGraph.__name__} ===")
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()

async def Invoke(graph, question):
    logging.info(f"\n=== {Invoke.__name__} ===")
    response = await graph.ainvoke({"question": question})
    logging.debug(f"Question: {question}")
    logging.debug(f"Response: {response["answer"]}")

async def Stream(graph, question):
    logging.info(f"\n=== {Stream.__name__} ===")
    logging.debug(f"Question: {question}")
    async for step in graph.astream(
        {"question": question}, stream_mode="updates"
    ):
        logging.debug(f"{step}\n\n----------------\n")

async def StreamTokens(graph, question):
    logging.info(f"\n=== {StreamTokens.__name__} ===")
    logging.debug(f"Question: {question}")
    async for message, metadata in graph.astream(
        {"question": question}, stream_mode="messages"
    ):
        logging.debug(message.content, end="|")

async def main():
    docs = vector_store.LoadDocuments("https://lilianweng.github.io/posts/2023-06-23-agent/")
    #subdocs = SplitDocuments(docs)
    #await IndexChunks(subdocs)
    config = RunnableConfig(run_name="RAG")
    graph: CompiledStateGraph = BuildGraph(config)
    show_graph(graph, "RAG")
    await Invoke(graph, "What is Task Decomposition?")
    await Stream(graph, "What is Task Decomposition?")
    await StreamTokens(graph, "What is Task Decomposition?")

if __name__ == "__main__":
    """
    image = graph.get_graph().draw_mermaid_png()
    # Save the PNG data to a file
    with open("/tmp/graph.png", "wb") as f:
        f.write(image)
    img = Image.open("/tmp/graph.png")
    img.show()
    """
    asyncio.run(main())