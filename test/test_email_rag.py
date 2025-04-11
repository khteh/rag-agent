import pytest, json
from datetime import datetime
from uuid_extensions import uuid7, uuid7str
from langchain_core.runnables import RunnableConfig
from src.common.State import EmailRAGState
from data.sample_emails import EMAILS
from src.models import ChatMessage
pytest_plugins = ('pytest_asyncio',)

@pytest.mark.asyncio(loop_scope="function")
async def test_email_escalate_safety_protocol(EmailRAGFixture):
    email_state = {
        "escalation_dollar_criteria": 100_000,
        "escalation_emails": ["brog@abc.com", "bigceo@company.com"],
    }
    config = RunnableConfig(run_name="Email escalate safety protocol", thread_id=uuid7str())
    """
    Should escalate because the workers are violating safety protocols by not wearing PPE.
    ai_message: type='ai' 
                content='The email from OSHA to Blue Ridge Construction indicates that there are several safety violations at the construction site in Dallas, TX. 
                        The violations include:\n\n1. Lack of fall protection: Workers on scaffolding above 10 feet were without required harnesses or other fall protection equipment.\n2. 
                        Unsafe scaffolding setup: Several scaffolding structures were noted as lacking secure base plates and bracing, creating potential collapse risks.\n3. Inadequate personal protective equipment (PPE): 
                        Multiple workers were found without proper PPE, including hard hats and safety glasses.\n\nTo rectify these violations, OSHA requires the following corrective actions:\n\n1. 
                        Install guardrails and fall arrest systems on all scaffolding over 10 feet.\n2. Conduct an inspection of all scaffolding structures and reinforce unstable sections.\n3. 
                        Ensure all workers on-site are provided with necessary PPE and conduct safety training on proper usage.\n\nThe deadline for compliance is November 10, 2025. Failure to comply may result in fines of up to $25,000 per violation. \n\n
                        This situation meets the escalation criteria because workers are explicitly violating safety protocols by not using required harnesses or other fall protection equipment, and not wearing proper PPE.' 
                tool_calls=[] tool_call_id=None run_id=None 
                original={'type': 'ai', 'data': {'content': 'The email from OSHA to Blue Ridge Construction indicates that there are several safety violations at the construction site in Dallas, TX. The violations include:\n\n1. Lack of fall protection: 
                        Workers on scaffolding above 10 feet were without required harnesses or other fall protection equipment.\n2. Unsafe scaffolding setup: Several scaffolding structures were noted as lacking secure base plates and bracing, creating potential collapse risks.\n3. Inadequate personal protective equipment (PPE): 
                        Multiple workers were found without proper PPE, including hard hats and safety glasses.\n\nTo rectify these violations, OSHA requires the following corrective actions:\n\n1. Install guardrails and fall arrest systems on all scaffolding over 10 feet.\n2. 
                        Conduct an inspection of all scaffolding structures and reinforce unstable sections.\n3. Ensure all workers on-site are provided with necessary PPE and conduct safety training on proper usage.\n\nThe deadline for compliance is November 10, 2025. Failure to comply may result in fines of up to $25,000 per violation. \n\n
                        This situation meets the escalation criteria because workers are explicitly violating safety protocols by not using required harnesses or other fall protection equipment, and not wearing proper PPE.', 
                	'additional_kwargs': {}, 
                	'response_metadata': {'model': 'llama3.3', 'created_at': '2025-04-05T07:27:29.787284732Z', 'done': True, 'done_reason': 'stop', 'total_duration': 368677640273, 'load_duration': 23849621, 'prompt_eval_count': 1229, 'prompt_eval_duration': 31958590162, 'eval_count': 242, 'eval_duration': 336691893036, 
                        'message': {'role': 'assistant', 'content': 'The email from OSHA to Blue Ridge Construction indicates that there are several safety violations at the construction site in Dallas, TX. The violations include:\n\n1. Lack of fall protection: Workers on scaffolding above 10 feet were without required harnesses or other fall protection equipment.\n2. 
                                Unsafe scaffolding setup: Several scaffolding structures were noted as lacking secure base plates and bracing, creating potential collapse risks.\n3. Inadequate personal protective equipment (PPE): Multiple workers were found without proper PPE, including hard hats and safety glasses.\n\nTo rectify these violations, OSHA requires the following corrective actions:\n\n1. 
                                Install guardrails and fall arrest systems on all scaffolding over 10 feet.\n2. Conduct an inspection of all scaffolding structures and reinforce unstable sections.\n3. Ensure all workers on-site are provided with necessary PPE and conduct safety training on proper usage.\n\nThe deadline for compliance is November 10, 2025. 
                                Failure to comply may result in fines of up to $25,000 per violation. \n\nThis situation meets the escalation criteria because workers are explicitly violating safety protocols by not using required harnesses or other fall protection equipment, and not wearing proper PPE.', 'images': None, 'tool_calls': None}}, 
                	'type': 'ai', 'name': None, 'id': 'run-25c482a2-57eb-40f5-a25a-4a6cdb771448-0', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 1229, 'output_tokens': 242, 'total_tokens': 1471}}}
    """
    lc_ai_message = await EmailRAGFixture.Chat("Workers explicitly violating safety protocols", EMAILS[0], email_state, config)
    #print(f"lc_ai_message: {lc_ai_message}")
    assert lc_ai_message
    ai_message = ChatMessage.from_langchain(lc_ai_message)
    #print(f"ai_message: {ai_message}")
    assert not ai_message.tool_calls
    assert ai_message.content
    assert "not using required harnesses or other fall protection equipment, and not wearing proper PPE" in ai_message.content

@pytest.mark.asyncio(loop_scope="function")
async def test_email_escalate_fire_safety_violation(EmailRAGFixture):
    email_state = {
        "escalation_dollar_criteria": 100_000,
        "escalation_emails": ["brog@abc.com", "bigceo@company.com"],
    }
    config = RunnableConfig(run_name="Email escalate fire safety", thread_id=uuid7str())
    """
    Should escalate because the workers are violating safety protocols by not wearing PPE.
    ai_message: type='ai' 
                content='The email from the City of Los Angeles Building and Safety Department indicates that there are building code violations at the Sunset Luxury Condominiums project site. The violations include exposed electrical wiring, insufficient fire extinguishers, and temporary support beams that do not meet load-bearing standards.\n\n
                        According to the escalation criteria, these issues pose an immediate risk of electrical or fire damage, which meets the criteria for escalation. Therefore, it is recommended that the necessary corrective actions be taken as soon as possible to address these violations and ensure the safety of the site.\n\n
                        The required changes include replacing or properly securing exposed wiring, installing additional fire extinguishers, and reinforcing or replacing temporary support beams. The deadline for compliance is October 31, 2025, and failure to comply may result in a stop-work order and additional fines.\n\n
                        It is recommended that the project team contact the Building and Safety Department at (555) 456-7890 or email inspections@lacity.gov to schedule a re-inspection and ensure that all necessary corrections are made.' 
                tool_calls=[] tool_call_id=None run_id=None 
                original={'type': 'ai', 'data': {'content': 'The email from the City of Los Angeles Building and Safety Department indicates that there are building code violations at the Sunset Luxury Condominiums project site. The violations include exposed electrical wiring, insufficient fire extinguishers, and temporary support beams that do not meet load-bearing standards.\n\n
                        According to the escalation criteria, these issues pose an immediate risk of electrical or fire damage, which meets the criteria for escalation. Therefore, it is recommended that the necessary corrective actions be taken as soon as possible to address these violations and ensure the safety of the site.\n\n
                        The required changes include replacing or properly securing exposed wiring, installing additional fire extinguishers, and reinforcing or replacing temporary support beams. The deadline for compliance is October 31, 2025, and failure to comply may result in a stop-work order and additional fines.\n\n
                        It is recommended that the project team contact the Building and Safety Department at (555) 456-7890 or email inspections@lacity.gov to schedule a re-inspection and ensure that all necessary corrections are made.', 
                    'additional_kwargs': {}, 
                    'response_metadata': {'model': 'llama3.3', 'created_at': '2025-04-06T07:57:05.968443296Z', 'done': True, 'done_reason': 'stop', 'total_duration': 318870900880, 'load_duration': 22874763, 'prompt_eval_count': 1135, 'prompt_eval_duration': 29028968866, 'eval_count': 207, 'eval_duration': 289815684097, 
                        'message': {'role': 'assistant', 'content': 'The email from the City of Los Angeles Building and Safety Department indicates that there are building code violations at the Sunset Luxury Condominiums project site. 
                                    The violations include exposed electrical wiring, insufficient fire extinguishers, and temporary support beams that do not meet load-bearing standards.\n\nAccording to the escalation criteria, these issues pose an immediate risk of electrical or fire damage, which meets the criteria for escalation. 
                                    Therefore, it is recommended that the necessary corrective actions be taken as soon as possible to address these violations and ensure the safety of the site.\n\nThe required changes include replacing or properly securing exposed wiring, installing additional fire extinguishers, and reinforcing or replacing temporary support beams. 
                                    The deadline for compliance is October 31, 2025, and failure to comply may result in a stop-work order and additional fines.\n\nIt is recommended that the project team contact the Building and Safety Department at (555) 456-7890 or email inspections@lacity.gov to schedule a re-inspection and ensure that all necessary corrections are made.', 'images': None, 'tool_calls': None}}, 
                    'type': 'ai', 'name': None, 'id': 'run-3aee0280-0793-4aa9-83ba-af73565a0e46-0', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 1135, 'output_tokens': 207, 'total_tokens': 1342}}}
    """
    lc_ai_message = await EmailRAGFixture.Chat("There's an immediate risk of electrical, water, or fire damage", EMAILS[3], email_state, config)
    #print(f"lc_ai_message: {lc_ai_message}")
    assert lc_ai_message
    ai_message = ChatMessage.from_langchain(lc_ai_message)
    print(f"ai_message: {ai_message}")
    assert not ai_message.tool_calls
    assert ai_message.content
    assert "building code violations at the Sunset Luxury Condominiums project site" in ai_message.content
    assert "immediate risk" in ai_message.content
    assert "fire damage" in ai_message.content

@pytest.mark.asyncio(loop_scope="function")
async def test_email_should_NOT_escalate(EmailRAGFixture):
    email_state = {
        "escalation_dollar_criteria": 100_000,
        "escalation_emails": ["brog@abc.com", "bigceo@company.com"],
    }
    config = RunnableConfig(run_name="Email should NOT escalate", thread_id=uuid7str())
    """
    EMAILS[0] doesn’t say anything about fire or water damage, and the maximum potential fine is less than $100,000.
    ai_message: type='ai' 
                content='The email from OSHA does not mention a risk of fire or water damage at the site. The identified safety violations are related to lack of fall protection, unsafe scaffolding setup, and inadequate personal protective equipment (PPE). 
                            Therefore, based on the provided escalation criteria, this issue does not require immediate escalation. However, it is crucial for Blue Ridge Construction to address these violations by the deadline of November 10, 2025, to avoid potential fines of up to $25,000 per violation.' 
                tool_calls=[] tool_call_id=None run_id=None 
                original={'type': 'ai', 'data': {'content': 'The email from OSHA does not mention a risk of fire or water damage at the site. The identified safety violations are related to lack of fall protection, unsafe scaffolding setup, and inadequate personal protective equipment (PPE). 
                            Therefore, based on the provided escalation criteria, this issue does not require immediate escalation. However, it is crucial for Blue Ridge Construction to address these violations by the deadline of November 10, 2025, to avoid potential fines of up to $25,000 per violation.', 
                        'additional_kwargs': {}, 
                        'response_metadata': {'model': 'llama3.3', 'created_at': '2025-04-05T07:46:47.7670311Z', 'done': True, 'done_reason': 'stop', 'total_duration': 169713928875, 'load_duration': 34967544, 'prompt_eval_count': 1244, 'prompt_eval_duration': 32563016753, 'eval_count': 102, 'eval_duration': 137111384910, 
                            'message': {'role': 'assistant', 'content': 'The email from OSHA does not mention a risk of fire or water damage at the site. The identified safety violations are related to lack of fall protection, unsafe scaffolding setup, and inadequate personal protective equipment (PPE). 
                                        Therefore, based on the provided escalation criteria, this issue does not require immediate escalation. However, it is crucial for Blue Ridge Construction to address these violations by the deadline of November 10, 2025, to avoid potential fines of up to $25,000 per violation.', 'images': None, 'tool_calls': None}}, 
                        'type': 'ai', 'name': None, 'id': 'run-0c14d6c5-b584-49c4-83f6-23fcf93726a5-0', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 1244, 'output_tokens': 102, 'total_tokens': 1346}}}
    """
    lc_ai_message = await EmailRAGFixture.Chat("There's a risk of fire or water damage at the site", EMAILS[0], email_state, config)
    #print(f"lc_ai_message: {lc_ai_message}")
    assert lc_ai_message
    ai_message = ChatMessage.from_langchain(lc_ai_message)
    #print(f"ai_message: {ai_message}")
    assert not ai_message.tool_calls
    assert ai_message.content
    assert "does not" in ai_message.content
    assert "November 10, 2025" in ai_message.content
    assert "fines of up to $25,000 per violation" in ai_message.content
