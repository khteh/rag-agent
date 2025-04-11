import pytest, sys, pytest_asyncio, logging, vertexai, os, sys
from datetime import datetime
from uuid_extensions import uuid7, uuid7str
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
    #vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    config = RunnableConfig(run_name="Email RAG", thread_id=uuid7str())
    from src.rag_agent.EmailRAG import EmailRAG
    rag = EmailRAG(config)
    await rag.CreateGraph()
    return rag

@pytest_asyncio.fixture(scope="function")
async def RAGAgentFixture():
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    #vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    config = RunnableConfig(run_name="RAG ReAct Agent", thread_id=uuid7str())
    from src.rag_agent.RAGAgent import RAGAgent
    rag = RAGAgent(config)
    await rag.CreateGraph()
    return rag

@pytest_asyncio.fixture(scope="function")
async def HealthcareRAGFixture() -> CompiledGraph:
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    #vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    config = RunnableConfig(run_name="Healthcare ReAct Agent", thread_id=uuid7str())
    from src.Healthcare.RAGAgent import RAGAgent
    rag = RAGAgent(config)
    await rag.CreateGraph()
    return rag

# https://docs.pytest.org/en/7.1.x/how-to/parametrize.html#basic-pytest-generate-tests-example