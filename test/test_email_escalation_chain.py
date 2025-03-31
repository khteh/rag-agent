import pytest, json
from data.sample_emails import EMAILS
from rag_agent.State import EmailRAGState

@pytest.mark.asyncio
@pytest.mark.skip(reason="https://github.com/langchain-ai/langchain/issues/30428")
async def test_needs_escalation_true(rag):
    state = {
         "message": EMAILS[0],
         "extract": None,
         "escalation_text_criteria": """Workers explicitly violating
                                        safety protocols""",
         "escalation_dollar_criteria": 100_000,
         "escalate": False,
         "escalation_emails": ["brog@abc.com", "bigceo@company.com"],
    }
    result: EmailRAGState = await rag.NeedsEscalation(state)
    assert result
    assert result["escalate"]

@pytest.mark.asyncio
@pytest.mark.skip(reason="https://github.com/langchain-ai/langchain/issues/30428")
async def test_needs_escalation_false(rag):
    state = {
         "message": EMAILS[0],
         "extract": None,
         "escalation_text_criteria": """There's a risk of fire or
                                        water damage at the site""",
         "escalation_dollar_criteria": 100_000,
         "escalate": False,
         "escalation_emails": ["brog@abc.com", "bigceo@company.com"],
    }
    result: EmailRAGState = await rag.NeedsEscalation(state)
    assert result
    assert not result["escalate"]
    