import pytest, json
from rag_agent.State import EmailRAGState
from data.sample_emails import EMAILS

@pytest.mark.asyncio
@pytest.mark.skip(reason="https://github.com/langchain-ai/langchain/discussions/30412")
async def test_email_parser_chain(rag):
    state = {
         "notice_message": EMAILS[0],
         "notice_email_extract": None,
         "escalation_text_criteria": """There's a risk of fire or
         water damage at the site""",
         "escalation_dollar_criteria": 100_000,
         "requires_escalation": False,
         "escalation_emails": ["brog@abc.com", "bigceo@company.com"],
    }
    result: EmailRAGState = await rag.ParseEmail(state)
    assert result
    assert result["notice_email_extract"]
    assert result["notice_email_extract"].entity_name == 'Occupational Safety and Health Administration (OSHA)'
    