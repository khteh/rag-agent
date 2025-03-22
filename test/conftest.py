import pytest, sys, asyncio, logging, vertexai, os, sys
from datetime import datetime
from dotenv import load_dotenv
from os.path import dirname, join, abspath
from langchain_core.runnables import RunnableConfig
from rag_agent.EmailRAG import EmailRAG
sys.path.insert(0, abspath(join(dirname(__file__), '../src')))
pytest_plugins = ('pytest_asyncio',)
load_dotenv()

@pytest.fixture(scope="function")
def rag():
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    config = RunnableConfig(run_name="Email RAG", thread_id=datetime.now())
    return EmailRAG(config)

def pytest_generate_tests(metafunc):
    """ called once per each test function """
    logging.info(f"\n=== {pytest_generate_tests.__name__} ===")
    os.environ['JWT_SECRET_KEY'] = 'pythonflaskrestapipostgres'