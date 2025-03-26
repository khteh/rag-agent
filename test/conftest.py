import pytest, sys, pytest_asyncio, logging, vertexai, os, sys
from datetime import datetime
from dotenv import load_dotenv
from os.path import dirname, join, abspath
from langchain_core.runnables import RunnableConfig
from langgraph.graph.graph import (
    END,
    START,
    CompiledGraph,
    Graph,
    Send,
)
sys.path.insert(0, abspath(join(dirname(__file__), '../src')))
pytest_plugins = ('pytest_asyncio',)
load_dotenv()

@pytest.fixture(scope="function")
def EmailRAG():
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    config = RunnableConfig(run_name="Email RAG", thread_id=datetime.now())
    from rag_agent.EmailRAG import EmailRAG
    return EmailRAG(config)

@pytest_asyncio.fixture(scope="function")
async def HealthcareRAG() -> CompiledGraph:
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    config = RunnableConfig(run_name="Healthcare ReAct Agent", thread_id=datetime.now())
    from Healthcare.RAGAgent import make_graph
    return await make_graph(config)

def pytest_generate_tests(metafunc):
    """ called once per each test function """
    logging.info(f"\n=== {pytest_generate_tests.__name__} ===")
    os.environ['JWT_SECRET_KEY'] = 'pythonflaskrestapipostgres'