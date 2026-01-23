import sys, pytest_asyncio, logging, sys
from uuid_extensions import uuid7, uuid7str
from os.path import dirname, join, abspath
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
sys.path.insert(0, abspath(join(dirname(__file__), '../src')))
pytest_plugins = ('pytest_asyncio',)

@pytest_asyncio.fixture(scope="function")
async def EmailRAGFixture():
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    #vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    config = RunnableConfig(run_name="Email RAG", configurable={"thread_id": uuid7str()})
    from src.rag_agent.EmailRAG import EmailRAG
    rag = EmailRAG(config)
    await rag.CreateGraph()
    return rag

@pytest_asyncio.fixture(scope="function")
async def GraphRAGFixture():
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    #vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    config = RunnableConfig(run_name="Checkpointed StateGraph RAG", configurable={"thread_id": uuid7str()})
    from src.rag_agent.GraphRAG import GraphRAG
    rag = GraphRAG(config)
    await rag.CreateGraph()
    return rag

@pytest_asyncio.fixture(scope="function")
async def RAGAgentFixture():
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    #vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    config = RunnableConfig(run_name="RAG Deep Agent", configurable={"thread_id": uuid7str()})
    from src.rag_agent.RAGAgent import RAGAgent
    rag = RAGAgent(config)
    await rag.CreateGraph()
    return rag

@pytest_asyncio.fixture(scope="function")
async def HealthcareRAGFixture() -> CompiledStateGraph:
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    #vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    config = RunnableConfig(run_name="Healthcare Deep Agent", configurable={"thread_id": uuid7str()})
    from src.Healthcare.RAGAgent import RAGAgent
    rag = RAGAgent(config)
    await rag.CreateGraph()
    return rag

# https://docs.pytest.org/en/7.1.x/how-to/parametrize.html#basic-pytest-generate-tests-example