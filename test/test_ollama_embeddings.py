import pytest
from langchain_ollama import OllamaEmbeddings
from src.config import config
pytest_plugins = ('pytest_asyncio',)

@pytest.mark.asyncio(loop_scope="function")
async def test_ollam_embeddings_vector_dimension():
    """
    https://python.langchain.com/api_reference/_modules/langchain_ollama/embeddings.html#OllamaEmbeddings
    https://huggingface.co/blog/matryoshka
    https://ollama.com/library/nomic-embed-text
    """
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL, base_url=config.BASE_URI, num_ctx=8192, num_gpu=1, temperature=0)
    result = await embeddings.aembed_documents(["Hello how are you doing"])
    dimension = (len(result[0])) # this should output 4096
    assert dimension <= 4096
