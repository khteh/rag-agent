import pytest, json
from datetime import datetime
from langchain_core.runnables import RunnableConfig
from rag_agent.State import EmailRAGState
from data.sample_emails import EMAILS
pytest_plugins = ('pytest_asyncio',)

@pytest.mark.asyncio(loop_scope="function")
async def test_email_should_escalate(EmailRAGFixture):
    email_state = {
        "escalation_dollar_criteria": 100_000,
        "escalation_emails": ["brog@abc.com", "bigceo@company.com"],
    }
    config = RunnableConfig(run_name="Email should escalate", thread_id=datetime.now())
    result = await EmailRAGFixture.Chat("\"Workers explicitly violating safety protocols\"", EMAILS[3], email_state, config)
    print(f"result: {result}")
    assert result

@pytest.mark.asyncio(loop_scope="function")
async def test_email_should_not_escalate(EmailRAGFixture):
    email_state = {
        "escalation_dollar_criteria": 100_000,
        "escalation_emails": ["brog@abc.com", "bigceo@company.com"],
    }
    config = RunnableConfig(run_name="Email should escalate", thread_id=datetime.now())
    result = await EmailRAGFixture.Chat("\"There's a risk of fire or water damage at the site\"", EMAILS[0], email_state, config)
    print(f"result: {result}")
    assert result
