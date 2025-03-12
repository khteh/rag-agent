import os, bs4, vertexai
from PIL import Image
from State import State
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
# https://python.langchain.com/docs/tutorials/rag/
load_dotenv()
print(f"PROJECT_ID: {os.environ.get("VERTEXAI_PROJECT_ID")}")
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

# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    # Define prompt for question-answering
    prompt = hub.pull("rlm/rag-prompt")
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

def BuildGraph():
    # Compile application and test
    print(f"\n=== {BuildGraph.__name__} ===")
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()

def Invoke(graph, question):
    print(f"\n=== {Invoke.__name__} ===")
    response = graph.invoke({"question": question})
    print(f"Question: {question}")
    print(f"Response: {response["answer"]}")

def Stream(graph, question):
    print(f"\n=== {Stream.__name__} ===")
    print(f"Question: {question}")
    for step in graph.stream(
        {"question": question}, stream_mode="updates"
    ):
        print(f"{step}\n\n----------------\n")

def StreamTokens(graph, question):
    print(f"\n=== {StreamTokens.__name__} ===")
    print(f"Question: {question}")
    for message, metadata in graph.stream(
        {"question": question}, stream_mode="messages"
    ):
        print(message.content, end="|")

if __name__ == "__main__":
    docs = LoadDocuments("https://lilianweng.github.io/posts/2023-06-23-agent/")
    subdocs = SplitDocuments(docs)
    IndexChunks(subdocs)
    graph = BuildGraph()
    image = graph.get_graph().draw_mermaid_png()
    # Save the PNG data to a file
    with open("/tmp/graph.png", "wb") as f:
        f.write(image)
    img = Image.open("/tmp/graph.png")
    img.show()        
    Invoke(graph, "What is Task Decomposition?")
    Stream(graph, "What is Task Decomposition?")
    StreamTokens(graph, "What is Task Decomposition?")