import os,pytest, re
from uuid_extensions import uuid7, uuid7str
from langchain_core.runnables import RunnableConfig
from src.models import ChatMessage
pytest_plugins = ('pytest_asyncio',)

@pytest.mark.asyncio(loop_scope="function")
async def test_ragagent_agent_blogs(RAGAgentFixture):
    config = RunnableConfig(run_name="RAG ReAct Agent", thread_id=uuid7str())
    lc_ai_message = await RAGAgentFixture.ChatAgent(config, [("human", "What is the standard method for Task Decomposition?"), ("human", "Once you get the answer, look up common extensions of that method.")])
    assert lc_ai_message
    ai_message = ChatMessage.from_langchain(lc_ai_message)
    #print(f"ai_message: {ai_message}")
    assert not ai_message.tool_calls
    assert ai_message.content
    print(f"test_ragagent_agent_blogs: {ai_message.content}")
    assert "Tree of Thoughts" in ai_message.content
    assert "Task-specific instructions" in ai_message.content
    assert "Human inputs" in ai_message.content
    assert "behavioral cloning" in ai_message.content
    assert "Algorithm Distillation" in ai_message.content
