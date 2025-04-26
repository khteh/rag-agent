import os,pytest, re
from uuid_extensions import uuid7, uuid7str
from langchain_core.runnables import RunnableConfig
from src.models import ChatMessage
pytest_plugins = ('pytest_asyncio',)

@pytest.mark.asyncio(loop_scope="function")
async def test_ragagent_agent_blogs(RAGAgentFixture):
    config = RunnableConfig(run_name=test_ragagent_agent_blogs.__name__, thread_id=uuid7str())
    lc_ai_message = await RAGAgentFixture.ChatAgent(config, ("What is task decomposition?\n" "What is the standard method for Task Decomposition?\n" "Once you get the answer, look up common extensions of that method."))
    assert lc_ai_message
    ai_message = ChatMessage.from_langchain(lc_ai_message)
    #print(f"ai_message: {ai_message}")
    assert not ai_message.tool_calls
    assert ai_message.content
    print(f"test_ragagent_agent_blogs: {ai_message.content}")
    assert "breaking down complex tasks" in ai_message.content
    assert "smaller, more manageable sub-tasks" in ai_message.content

@pytest.mark.asyncio(loop_scope="function")
async def test_ragagent_mlflow_blogs(RAGAgentFixture):
    config = RunnableConfig(run_name=test_ragagent_mlflow_blogs.__name__, thread_id=uuid7str())
    lc_ai_message = await RAGAgentFixture.ChatAgent(config, ("What is MLFlow?\n""Where can you run it?"))
    assert lc_ai_message
    ai_message = ChatMessage.from_langchain(lc_ai_message)
    #print(f"ai_message: {ai_message}")
    assert not ai_message.tool_calls
    assert ai_message.content
    print(f"test_ragagent_mlflow_blogs: {ai_message.content}")
    assert "machine learning" in ai_message.content
    assert "track, manage, and deploy" in ai_message.content
    assert "Local Machine" in ai_message.content
    assert "Cloud Platforms" in ai_message.content
    assert "Docker Containers" in ai_message.content
    assert "Kubernetes Clusters" in ai_message.content
    assert "MLFlow UI" in ai_message.content
    assert "MLFlow CLI" in ai_message.content
    assert "Jupyter Notebooks" in ai_message.content
