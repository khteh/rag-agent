from dotenv import load_dotenv
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
embeddings = VertexAIEmbeddings(model="text-embedding-005")
global vector_store
# https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
vector_store = InMemoryVectorStore(embeddings)