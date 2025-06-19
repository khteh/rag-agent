from pprint import pprint

def test_value_error():
    value_error = ValueError("Found AIMessages with tool_calls that do not have a corresponding ToolMessage. Here are the first few of those tool calls: [{'name': 'HealthcareCypher', 'args': {'query': 'Which physician has treated the most patients covered by Cigna?'}, 'id': 'c6057e7a-517e-45f4-b07a-799ada7e2c1c', 'type': 'tool_call'}].\n\nEvery tool call (LLM requesting to call a tool) in the message history MUST have a corresponding ToolMessage (result of a tool invocation to return to the LLM) - this is required by most LLM providers.\nFor troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/INVALID_CHAT_HISTORY")
    pprint(f"args: {value_error.args}")
    assert "tool_call" in value_error.args[0]