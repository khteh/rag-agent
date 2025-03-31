import pytest, json
from rag_agent.State import EmailRAGState
from data.sample_emails import EMAILS

@pytest.mark.asyncio
@pytest.mark.skip(reason="https://github.com/langchain-ai/langchain/discussions/30412")
async def test_email_parser_chain(EmailRAG):
    state = {
         "message": EMAILS[0],
         "extract": None,
         "escalation_text_criteria": """There's a risk of fire or
         water damage at the site""",
         "escalation_dollar_criteria": 100_000,
         "escalate": False,
         "escalation_emails": ["brog@abc.com", "bigceo@company.com"],
    }
    result: EmailRAGState = await EmailRAG.ParseEmail(state)
    assert result
    assert result["extract"]
    assert result["extract"].date_str
    assert result["extract"].entity_name == 'Occupational Safety and Health Administration (OSHA)'
    assert result["extract"].entity_phone
    assert result["extract"].entity_phone == "(555) 123-4567"
    assert result["extract"].entity_email
    assert result["extract"].entity_email == "compliance.osha@osha.gov"
    assert result["extract"].project_id
    assert result["extract"].project_id == 111232345
    assert result["extract"].site_location
    assert result["extract"].site_location == "123 Main Street, Dallas, TX"
    assert result["extract"].escalate
    assert result["extract"].violation_type
    assert result["extract"].required_changes
    assert result["extract"].compliance_deadline
    assert result["extract"].max_potential_fine
    assert result["extract"].max_potential_fine == 25000.0