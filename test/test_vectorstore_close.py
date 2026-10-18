import importlib, pytest
from unittest.mock import AsyncMock, MagicMock
from src.Infrastructure.VectorStore import VectorStore

"""
VectorStore.close() releases the connections held by its PGEngine. The psycopg pool belongs to the caller and must not be closed by it.
No database is needed: neither the PGEngine nor the agents connect until they are used.
"""
def make_vectorstore(pool=None) -> VectorStore:
    return VectorStore(db_pool=pool or MagicMock())

def make_agent(module: str, name: str, **attributes):
    # Skip __init__ so that no LLM client or connection pool gets created.
    agent = object.__new__(getattr(importlib.import_module(module), name))
    agent._closed = False
    agent._self_managed_db_pool = False
    agent._db_pool = AsyncMock()
    agent._vectorStore = AsyncMock()
    for key, value in attributes.items():
        setattr(agent, key, value)
    return agent

async def test_close_disposes_the_real_engine():
    vectorstore = make_vectorstore()
    assert vectorstore._pg_engine is not None
    await vectorstore.close() # PGEngine.close() runs on the engine's own background loop
    assert vectorstore._pg_engine is None

async def test_close_awaits_engine_close_only_once():
    vectorstore = make_vectorstore()
    engine = AsyncMock()
    vectorstore._pg_engine = engine
    await vectorstore.close()
    await vectorstore.close()
    engine.close.assert_awaited_once()

async def test_close_leaves_the_callers_pool_alone():
    pool = AsyncMock()
    await make_vectorstore(pool).close()
    pool.close.assert_not_awaited()

async def test_close_swallows_engine_errors():
    vectorstore = make_vectorstore()
    vectorstore._pg_engine = AsyncMock()
    vectorstore._pg_engine.close.side_effect = RuntimeError("boom")
    await vectorstore.close() # Cleanup must never raise
    assert vectorstore._pg_engine is None

@pytest.mark.parametrize("module, name", [("src.rag_agent.RAGAgent", "RAGAgent"), ("src.EmailRAG.EmailRAG", "EmailRAG")])
async def test_agent_cleanup_closes_the_vectorstore_it_created(module, name):
    agent = make_agent(module, name, _self_managed_db_pool = True)
    await agent._Cleanup()
    agent._vectorStore.close.assert_awaited_once()
    agent._db_pool.close.assert_awaited_once()
    assert agent._closed

@pytest.mark.parametrize("module, name", [("src.rag_agent.RAGAgent", "RAGAgent"), ("src.EmailRAG.EmailRAG", "EmailRAG")])
async def test_agent_cleanup_leaves_a_provided_pool_open(module, name):
    # The hypercorn web app case: main.py creates the pool, passes it in and closes it itself.
    agent = make_agent(module, name, _self_managed_db_pool = False)
    await agent._Cleanup()
    agent._vectorStore.close.assert_awaited_once()
    agent._db_pool.close.assert_not_awaited()

@pytest.mark.parametrize("module, name", [("src.rag_agent.RAGAgent", "RAGAgent"), ("src.EmailRAG.EmailRAG", "EmailRAG")])
async def test_agent_cleanup_without_a_vectorstore(module, name):
    # CreateGraph() may fail before a VectorStore exists.
    agent = make_agent(module, name, _vectorStore = None)
    await agent._Cleanup()
    assert agent._closed

async def test_healthcare_agent_closes_only_a_vectorstore_it_created():
    module, name = "src.Healthcare.RAGAgent", "RAGAgent"
    provided = make_agent(module, name, _self_managed_vectorStore = False) # Passed in by rag_agent.RAGAgent, which owns it
    await provided._Cleanup()
    provided._vectorStore.close.assert_not_awaited()
    created = make_agent(module, name, _self_managed_vectorStore = True)
    await created._Cleanup()
    created._vectorStore.close.assert_awaited_once()