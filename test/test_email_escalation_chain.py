import pytest, json
from datetime import datetime
from uuid_extensions import uuid7, uuid7str
from langchain_core.runnables import RunnableConfig
from data.sample_emails import EMAILS
from src.common.State import EmailRAGState

@pytest.mark.asyncio(loop_scope="function")
async def test_needs_escalation_true(EmailRAGFixture):
    state = {
         "email": EMAILS[0],
         "extract": None,
         "escalation_text_criteria": "Workers explicitly violating safety protocols",
         "escalation_dollar_criteria": 100_000,
         "escalate": False,
         "escalation_emails": ["brog@abc.com", "bigceo@company.com"],
    }
    config = RunnableConfig(run_name=test_needs_escalation_true.__name__, configurable={"thread_id": uuid7str()})
    result: EmailRAGState = await EmailRAGFixture.NeedsEscalation(state, config)
    assert result
    assert result["escalate"]

@pytest.mark.asyncio(loop_scope="function")
async def test_needs_escalation_false(EmailRAGFixture):
    """
    requires_escalation is set to False because EMAILS[0] doesn’t say anything about fire or water damage, and the maximum potential fine is less than $100,000.
    """
    state = {
         "email": EMAILS[0],
         "extract": None,
         "escalation_text_criteria": "There's a risk of fire or water damage at the site",
         "escalation_dollar_criteria": 100_000,
         "escalate": False,
         "escalation_emails": ["brog@abc.com", "bigceo@company.com"],
    }
    config = RunnableConfig(run_name=test_needs_escalation_false.__name__, configurable={"thread_id": uuid7str()})
    result: EmailRAGState = await EmailRAGFixture.NeedsEscalation(state, config)
    assert result
    assert not result["escalate"]
    