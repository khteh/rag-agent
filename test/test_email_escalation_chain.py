import pytest, json
from datetime import datetime
from langchain_core.runnables import RunnableConfig
from data.sample_emails import EMAILS
from rag_agent.State import EmailRAGState

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
    config = RunnableConfig(run_name="Email RAG Test", thread_id=datetime.now())
    result: EmailRAGState = await EmailRAGFixture.NeedsEscalation(state, config)
    print(f"result: {result}")
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
    config = RunnableConfig(run_name="Email RAG Test", thread_id=datetime.now())
    result: EmailRAGState = await EmailRAGFixture.NeedsEscalation(state, config)
    print(f"result: {result}")
    assert result
    assert result["extract"].max_potential_fine
    assert result["extract"].max_potential_fine == 25000.0    
    assert not result["escalate"]
    