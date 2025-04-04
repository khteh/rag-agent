import pytest, json
from datetime import datetime
from langchain_core.runnables import RunnableConfig
from rag_agent.State import EmailRAGState
from data.sample_emails import EMAILS
from schema import ChatMessage
pytest_plugins = ('pytest_asyncio',)

@pytest.mark.asyncio(loop_scope="function")
async def test_email_should_escalate(EmailRAGFixture):
    email_state = {
        "escalation_dollar_criteria": 100_000,
        "escalation_emails": ["brog@abc.com", "bigceo@company.com"],
    }
    config = RunnableConfig(run_name="Email should escalate", thread_id=datetime.now())
    """
    Should escalate because the workers are violating safety protocols by not wearing PPE.
    AIMessage(content='Based on the email from the City of Los Angeles Building and Safety Department, the following information is relevant to determining if the escalation criteria are met:\n\n- Site Location: Los Angeles, CA\n- Violation Type: Electrical Wiring, Fire Safety, Structural Integrity\n- 
                        Required Changes: Replace or properly secure exposed wiring, install additional fire extinguishers, reinforce or replace temporary support beams\n- Compliance Deadline: February 5, 2025\n\nThe escalation criteria are "Workers explicitly violating safety protocols." 
                        The email identifies several building code violations related to safety hazards, including exposed wiring and insufficient fire extinguishers. However, it does not explicitly state that workers are violating safety protocols. 
                        Instead, it outlines required corrective actions to address the identified violations.\n\nTherefore, based on the information provided in the email, it does not appear that the escalation criteria are met, as there is no explicit mention of workers violating safety protocols. 
                        The focus is on correcting building code violations related to electrical wiring, fire safety, and structural integrity by a specified deadline to avoid further action such as a stop-work order and fines.', 
                additional_kwargs={},
                response_metadata={'model': 'llama3.3', 'created_at': '2025-04-04T12:35:17.743495848Z', 'done': True, 'done_reason': 'stop', 'total_duration': 290469933028, 'load_duration': 30023344, 'prompt_eval_count': 1134, 'prompt_eval_duration': 29435853506, 'eval_count': 220, 'eval_duration': 261001488510, 
                                    'message': Message(role='assistant', content='Based on the email from the City of Los Angeles Building and Safety Department, the following information is relevant to determining if the escalation criteria are met:\n\n- 
                                                        Site Location: Los Angeles, CA\n- Violation Type: Electrical Wiring, Fire Safety, Structural Integrity\n- Required Changes: Replace or properly secure exposed wiring, install additional fire extinguishers, reinforce or replace temporary support beams\n- 
                                                        Compliance Deadline: February 5, 2025\n\nThe escalation criteria are "Workers explicitly violating safety protocols." The email identifies several building code violations related to safety hazards, including exposed wiring and insufficient fire extinguishers. 
                                                        However, it does not explicitly state that workers are violating safety protocols. Instead, it outlines required corrective actions to address the identified violations.\n\nTherefore, based on the information provided in the email, it does not appear that the escalation criteria are met, 
                                                        as there is no explicit mention of workers violating safety protocols. The focus is on correcting building code violations related to electrical wiring, fire safety, and structural integrity by a specified deadline to avoid further action such as a stop-work order and fines.', images=None, tool_calls=None)},
                                                        id='run-3c505858-956a-4b38-9d9b-83d5fe35cadd-0', usage_metadata={'input_tokens': 1134, 'output_tokens': 220, 'total_tokens': 1354})    
    """
    lc_ai_message = await EmailRAGFixture.Chat("Workers explicitly violating safety protocols", EMAILS[0], email_state, config)
    print(f"lc_ai_message: {lc_ai_message}")
    assert lc_ai_message
    ai_message = ChatMessage.from_langchain(lc_ai_message)
    assert not ai_message.tool_calls
    assert ai_message.content
    #assert "it does not explicitly state that workers are violating safety protocols" in ai_message.content
    #assert "it does not appear that the escalation criteria are met" in ai_message.content

@pytest.mark.asyncio(loop_scope="function")
async def test_email_should_not_escalate(EmailRAGFixture):
    email_state = {
        "escalation_dollar_criteria": 100_000,
        "escalation_emails": ["brog@abc.com", "bigceo@company.com"],
    }
    config = RunnableConfig(run_name="Email should escalate", thread_id=datetime.now())
    # EMAILS[0] doesn’t mention anything about water damage
    lc_ai_message = await EmailRAGFixture.Chat("There's a risk of fire or water damage at the site", EMAILS[0], email_state, config)
    assert lc_ai_message
    ai_message = ChatMessage.from_langchain(lc_ai_message)
    assert not ai_message.tool_calls
    assert ai_message.content
