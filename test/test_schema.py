from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, ToolCall
from models import ChatMessage

def test_messages_to_langchain():
    human_message = ChatMessage(type="human", content="Hello, world!")
    lc_message = human_message.to_langchain()
    assert isinstance(lc_message, HumanMessage)
    assert lc_message.type == "human"
    assert lc_message.content == "Hello, world!"

def test_messages_from_langchain():
    lc_human_message = HumanMessage(content="Hello, world!")
    human_message = ChatMessage.from_langchain(lc_human_message)
    assert human_message.type == "human"
    assert human_message.content == "Hello, world!"
    assert lc_human_message == human_message.to_langchain()

    lc_ai_message = AIMessage(content="Hello, world!")
    ai_message = ChatMessage.from_langchain(lc_ai_message)
    assert ai_message.type == "ai"
    assert ai_message.content == "Hello, world!"
    assert lc_ai_message == ai_message.to_langchain()

    lc_tool_message = ToolMessage(content="Hello, world!", tool_call_id="123")
    tool_message = ChatMessage.from_langchain(lc_tool_message)
    assert tool_message.type == "tool"
    assert tool_message.content == "Hello, world!"
    assert tool_message.tool_call_id == "123"
    assert lc_tool_message == tool_message.to_langchain()

    lc_system_message = SystemMessage(content="Hello, world!")
    try:
        _ = ChatMessage.from_langchain(lc_system_message)
    except ValueError as e:
        assert str(e) == "Unsupported message type: SystemMessage"

def test_messages_tool_calls():
    tool_call = ToolCall(name="test_tool", args={"x": 1, "y": 2}, id="call_Jja7")
    lc_ai_message = AIMessage(content="", tool_calls=[tool_call])
    ai_message = ChatMessage.from_langchain(lc_ai_message)
    assert ai_message.tool_calls[0]["id"] == "call_Jja7"
    assert ai_message.tool_calls[0]["name"] == "test_tool"
    assert ai_message.tool_calls[0]["args"] == {"x": 1, "y": 2}
    assert lc_ai_message == ai_message.to_langchain()

def test_ai_message_with_tool_call():
    lc_ai_message = AIMessage(content='', additional_kwargs={}, response_metadata={'model': 'llama3.3', 'created_at': '2025-06-19T02:29:14.245123244Z', 'done': True, 'done_reason': 'stop', 'total_duration': 84698753836, 'load_duration': 18683522239, 'prompt_eval_count': 682, 'prompt_eval_duration': 21478345328, 'eval_count': 31, 'eval_duration': 44535435524, 'model_name': 'gpt-oss'}, name='RAG Deep Agent', id='run--aa83bac8-f035-4d8d-842a-dadb6d89049e-0', tool_calls=[{'name': 'HealthcareCypher', 'args': {'query': 'Which physician has treated the most patients covered by Cigna?'}, 'id': 'c6057e7a-517e-45f4-b07a-799ada7e2c1c', 'type': 'tool_call'}], usage_metadata={'input_tokens': 682, 'output_tokens': 31, 'total_tokens': 713})