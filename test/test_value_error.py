import json
from pprint import pprint
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    ToolCall,
    message_to_dict,
    messages_from_dict,
)
from src.models.schema import ChatMessage, UserInput, StreamInput

def test_value_error():
    value_error = ValueError("Found AIMessages with tool_calls that do not have a corresponding ToolMessage. Here are the first few of those tool calls: [{'name': 'HealthcareCypher', 'args': {'query': 'Which physician has treated the most patients covered by Cigna?'}, 'id': 'c6057e7a-517e-45f4-b07a-799ada7e2c1c', 'type': 'tool_call'}].\n\nEvery tool call (LLM requesting to call a tool) in the message history MUST have a corresponding ToolMessage (result of a tool invocation to return to the LLM) - this is required by most LLM providers.\nFor troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/INVALID_CHAT_HISTORY")
    pprint(f"args: {value_error.args}")
    assert "tool_call" in value_error.args[0]

def test_message():
    msg = {'messages': [ToolMessage(content="I don't know!", tool_call_id=123),
                        AIMessage(content='Hello World!!!', additional_kwargs={}, response_metadata={'model': 'gpt-oss', 'created_at': '2026-01-28T04:46:25.230706269Z', 'done': True, 'done_reason': 'stop', 'total_duration': 199014737212, 'load_duration': 307089957, 'prompt_eval_count': 8192, 'prompt_eval_duration': 35284390138, 'eval_count': 1187, 'eval_duration': 162487149022, 'logprobs': None, 'model_name': 'gpt-oss', 'model_provider': 'ollama'}, id='lc_run--019c02e9-8c61-7e73-a9b0-94e5c3f46d97-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 8192, 'output_tokens': 1187, 'total_tokens': 9379})]}
    new_messages = []
    #logging.debug(f"s: {s}")
    #for _, state in msg.items():
    print(f"state: {msg}")
    if "messages" in msg:
        new_messages.extend(msg["messages"])
    assert 2 == len(new_messages)
    tool_message = ChatMessage.from_langchain(new_messages[0])
    assert not tool_message.tool_calls
    assert "tool" == tool_message.type
    assert "I don't know!" == tool_message.content

    ai_message = ChatMessage.from_langchain(new_messages[1])
    assert not ai_message.tool_calls
    assert "ai" == ai_message.type
    assert ai_message.content
    assert "Hello World!!!" == ai_message.content