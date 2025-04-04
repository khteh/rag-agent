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
    lc_ai_message: content='The email from OSHA indicates that there are several safety violations at the construction site located at 123 Main Street in Dallas, TX. 
                        The violations include:\n\n1. Lack of fall protection: Workers on scaffolding above 10 feet were without required harnesses or other fall protection equipment.\n2. 
                        Unsafe scaffolding setup: Several scaffolding structures were noted as lacking secure base plates and bracing, creating potential collapse risks.\n3. 
                        Inadequate personal protective equipment (PPE): Multiple workers were found without proper PPE, including hard hats and safety glasses.\n\nTo rectify these violations, the following corrective actions are required:\n\n1. 
                        Install guardrails and fall arrest systems on all scaffolding over 10 feet.\n2. Conduct an inspection of all scaffolding structures and reinforce unstable sections.\n3. 
                        Ensure all workers on-site are provided with necessary PPE and conduct safety training on proper usage.\n\nThe deadline for compliance is November 10, 2025. Failure to comply may result in fines of up to $25,000 per violation. \n\n
                        This situation meets the escalation criteria because workers are explicitly violating safety protocols by not using required harnesses or other fall protection equipment, and by not wearing proper PPE. 
                        Therefore, immediate attention is necessary to ensure the safety of all workers on-site.' 
                    additional_kwargs={} 
                    response_metadata={'model': 'llama3.3', 'created_at': '2025-04-04T13:37:17.629453608Z', 'done': True, 'done_reason': 'stop', 'total_duration': 336589157511, 'load_duration': 14513823, 'prompt_eval_count': 1255, 'prompt_eval_duration': 30842738181, 'eval_count': 260, 'eval_duration': 305729677041, 
                    'message': Message(role='assistant', content='The email from OSHA indicates that there are several safety violations at the construction site located at 123 Main Street in Dallas, TX. The violations include:\n\n1. 
                        Lack of fall protection: Workers on scaffolding above 10 feet were without required harnesses or other fall protection equipment.\n2. 
                        Unsafe scaffolding setup: Several scaffolding structures were noted as lacking secure base plates and bracing, creating potential collapse risks.\n3. Inadequate personal protective equipment (PPE): 
                        Multiple workers were found without proper PPE, including hard hats and safety glasses.\n\nTo rectify these violations, the following corrective actions are required:\n\n1. Install guardrails and fall arrest systems on all scaffolding over 10 feet.\n2. 
                        Conduct an inspection of all scaffolding structures and reinforce unstable sections.\n3. Ensure all workers on-site are provided with necessary PPE and conduct safety training on proper usage.\n\nThe deadline for compliance is November 10, 2025. 
                        Failure to comply may result in fines of up to $25,000 per violation. \n\nThis situation meets the escalation criteria because workers are explicitly violating safety protocols by not using required harnesses or other fall protection equipment, and by not wearing proper PPE. 
                        Therefore, immediate attention is necessary to ensure the safety of all workers on-site.', images=None, tool_calls=None)}
                    id='run-b3f365ed-8bd9-402e-b4f2-2020a70cebf7-0' usage_metadata={'input_tokens': 1255, 'output_tokens': 260, 'total_tokens': 1515}
    """
    lc_ai_message = await EmailRAGFixture.Chat("Workers explicitly violating safety protocols", EMAILS[0], email_state, config)
    print(f"lc_ai_message: {lc_ai_message}")
    assert lc_ai_message
    ai_message = ChatMessage.from_langchain(lc_ai_message)
    assert not ai_message.tool_calls
    assert ai_message.content
    assert "several safety violations at the construction site located at 123 Main Street in Dallas, TX." in ai_message.content
    assert "This situation meets the escalation criteria because workers are explicitly violating safety protocols by not using required harnesses or other fall protection equipment, and by not wearing proper PPE." in ai_message.content

@pytest.mark.asyncio(loop_scope="function")
async def test_email_should_not_escalate(EmailRAGFixture):
    email_state = {
        "escalation_dollar_criteria": 100_000,
        "escalation_emails": ["brog@abc.com", "bigceo@company.com"],
    }
    config = RunnableConfig(run_name="Email should escalate", thread_id=datetime.now())
    """
    EMAILS[0] doesn’t say anything about fire or water damage, and the maximum potential fine is less than $100,000.
    lc_ai_message: content='The email from OSHA does not mention a risk of fire or water damage at the site. The identified safety violations are related to lack of fall protection, unsafe scaffolding setup, and inadequate personal protective equipment (PPE). 
                        Therefore, based on the provided escalation criteria, this issue does not need to be escalated.' 
                        additional_kwargs={} 
            response_metadata={'model': 'llama3.3', 'created_at': '2025-04-04T15:49:12.383353674Z', 'done': True, 'done_reason': 'stop', 'total_duration': 108626867720, 'load_duration': 28964426, 'prompt_eval_count': 1270, 'prompt_eval_duration': 29792799649, 'eval_count': 64, 'eval_duration': 78802620013, 
            'message': Message(role='assistant', content='The email from OSHA does not mention a risk of fire or water damage at the site. The identified safety violations are related to lack of fall protection, unsafe scaffolding setup, and inadequate personal protective equipment (PPE). 
                            Therefore, based on the provided escalation criteria, this issue does not need to be escalated.', images=None, tool_calls=None)} 
            id='run-e04cb6c5-ce95-4d60-bf4b-d65887e00e73-0' usage_metadata={'input_tokens': 1270, 'output_tokens': 64, 'total_tokens': 1334}
    """
    lc_ai_message = await EmailRAGFixture.Chat("There's a risk of fire or water damage at the site", EMAILS[0], email_state, config)
    print(f"lc_ai_message: {lc_ai_message}")
    assert lc_ai_message
    ai_message = ChatMessage.from_langchain(lc_ai_message)
    assert not ai_message.tool_calls
    assert ai_message.content
    assert "this issue does not need to be escalated" in ai_message.content
