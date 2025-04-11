import pytest, json
from datetime import datetime, date
from uuid_extensions import uuid7, uuid7str
from langchain_core.runnables import RunnableConfig
from src.common.State import EmailRAGState
from data.sample_emails import EMAILS

@pytest.mark.asyncio(loop_scope="function")
async def test_email_parser_chain_email0(EmailRAGFixture):
    state = {
         "email": EMAILS[0],
         "extract": None,
         "escalation_text_criteria": "There's a risk of fire or water damage at the site",
         "escalation_dollar_criteria": 100_000,
         "escalate": False,
         "escalation_emails": ["brog@abc.com", "bigceo@company.com"],
    }
    config = RunnableConfig(run_name=test_email_parser_chain_email0.__name__, thread_id=uuid7str())
    result: EmailRAGState = await EmailRAGFixture.ParseEmail(state, config)
    #print(f"extract: {result['extract']}")
    assert result
    assert result["extract"]
    assert result["extract"].date_str
    assert result["extract"].date_of_email
    assert result["extract"].date_of_email == date(2025, 4, 2)
    assert result["extract"].name == 'Occupational Safety and Health Administration (OSHA)'
    if result["extract"].phone:
     assert result["extract"].phone == "(555) 123-4567"
    assert result["extract"].email
    assert result["extract"].email == "compliance.osha@osha.gov"
    assert result["extract"].project_id
    assert result["extract"].project_id == 111232345
    assert result["extract"].site_location
    assert result["extract"].site_location == "123 Main Street, Dallas, TX"
    assert result["extract"].violation_type
    assert result["extract"].required_changes
    assert result["extract"].compliance_deadline
    assert result["extract"].compliance_deadline == date(2025, 11, 10)
    assert result["extract"].max_potential_fine
    assert result["extract"].max_potential_fine == 25000.0

@pytest.mark.asyncio(loop_scope="function")
async def test_email_parser_chain_email3(EmailRAGFixture):
    state = {
         "email": EMAILS[3],
         "extract": None,
         "escalation_text_criteria": "There's an immediate risk of electrical, water, or fire damage",
         "escalation_dollar_criteria": 100_000,
         "escalate": False,
         "escalation_emails": ["brog@abc.com", "bigceo@company.com"],
    }
    config = RunnableConfig(run_name=test_email_parser_chain_email3.__name__, thread_id=uuid7str())
    result: EmailRAGState = await EmailRAGFixture.ParseEmail(state, config)
    #print(f"extract: {result['extract']}")
    assert result
    assert result["extract"]
    assert result["extract"].date_str
    assert result["extract"].date_of_email
    assert result["extract"].date_of_email == date(2025, 4, 3)
    assert result["extract"].name == 'City of Los Angeles Building and Safety Department'
    #assert result["extract"].phone
    if result["extract"].phone:
        assert result["extract"].phone == "(555) 123-4567"
    assert result["extract"].email
    assert result["extract"].email == "inspections@lacity.gov"
    if result["extract"].project_id:
        assert result["extract"].project_id == 345678123
    assert result["extract"].site_location
    assert result["extract"].site_location == "456 Sunset Boulevard, Los Angeles, CA"
    assert result["extract"].violation_type
    assert result["extract"].required_changes
    assert result["extract"].compliance_deadline
    assert result["extract"].compliance_deadline == date(2025, 10, 31)
    assert not result["extract"].max_potential_fine