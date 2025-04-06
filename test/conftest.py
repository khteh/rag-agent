import pytest, sys, pytest_asyncio, logging, vertexai, os, sys
from datetime import datetime
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

@pytest_asyncio.fixture(scope="function")
async def EmailRAGFixture():
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    ##vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    config = RunnableConfig(run_name="Email RAG", thread_id=datetime.now())
    from rag_agent.EmailRAG import EmailRAG
    rag = EmailRAG(config)
    await rag.CreateGraph()
    return rag

@pytest_asyncio.fixture(scope="function")
async def HealthcareRAGFixture() -> CompiledGraph:
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    ##vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    config = RunnableConfig(run_name="Healthcare ReAct Agent", thread_id=datetime.now())
    from Healthcare.RAGAgent import make_graph
    return await make_graph(config)

# https://docs.pytest.org/en/7.1.x/how-to/parametrize.html#basic-pytest-generate-tests-example