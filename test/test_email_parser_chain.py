import pytest, json
from datetime import datetime
from langchain_core.runnables import RunnableConfig
from rag_agent.State import EmailRAGState
from data.sample_emails import EMAILS

@pytest.mark.asyncio(loop_scope="function")
async def test_email_parser_chain(EmailRAGFixture):
    state = {
         "email": EMAILS[0],
         "extract": None,
         "escalation_text_criteria": "\"There's a risk of fire or water damage at the site\"",
         "escalation_dollar_criteria": 100_000,
         "escalate": False,
         "escalation_emails": ["brog@abc.com", "bigceo@company.com"],
    }
    config = RunnableConfig(run_name="Email RAG Test", thread_id=datetime.now())
    result: EmailRAGState = await EmailRAGFixture.ParseEmail(state, config)
    print(f"extract: {result['extract']}")
    assert result
    assert result["extract"]
    assert result["extract"].date_str
    assert result["extract"].name == 'Occupational Safety and Health Administration (OSHA)'
    assert result["extract"].phone
    assert result["extract"].phone == "(555) 123-4567"
    assert result["extract"].email
    assert result["extract"].email == "compliance.osha@osha.gov"
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