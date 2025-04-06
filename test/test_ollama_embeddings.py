import pytest
from langchain_ollama import OllamaEmbeddings
from src.config import config
pytest_plugins = ('pytest_asyncio',)

@pytest.mark.asyncio(loop_scope="function")
async def test_ollam_embeddings_vector_dimension():
    embeddings = OllamaEmbeddings(model="llama3.3", base_url=config.OLLAMA_URI, num_ctx=4096, num_gpu=1, temperature=0, top_k=10)
    result = await embeddings.aembed_documents(["Hello how are you doing"])
    dimension = (len(result[0])) # this should output 4096
    print(f"dimension: {dimension}")
    #assert 4096 == dimension
