import pytest, json
from datetime import datetime
from uuid_extensions import uuid7, uuid7str
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import convert_to_messages
from src.models import ChatMessage
from src.common.State import EmailRAGState

@pytest.mark.asyncio(loop_scope="function")
async def test_generate_query_or_respond(GraphRAGFixture):
    config = RunnableConfig(run_name=test_generate_query_or_respond.__name__, configurable={"thread_id": uuid7str()})
    input = {"messages": [{"role": "user", "content": "Hello, who are you?"}]}
    response = await GraphRAGFixture.Agent(input, config)
    lc_ai_message = response["messages"][-1]
    assert lc_ai_message
    ai_message = ChatMessage.from_langchain(lc_ai_message)
    print(f"{test_generate_query_or_respond.__name__} response: {len(ai_message.content)}, {response["messages"][-1].pretty_print()}")
    assert len(ai_message.content)

@pytest.mark.asyncio(loop_scope="function")
async def test_should_rewrite_question(GraphRAGFixture):
    config = RunnableConfig(run_name=test_should_rewrite_question.__name__, configurable={"thread_id": uuid7str()})
    input = {
        "messages": convert_to_messages(
            [
                {
                    "role": "user",
                    "content": "What does Lilian Weng say about types of reward hacking?",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "1",
                            "name": "retrieve_blog_posts",
                            "args": {"query": "types of reward hacking"},
                        }
                    ],
                },
                {"role": "tool", "content": "meow", "tool_call_id": "1"},
            ]
        )
    }
    assert "Rewrite" == await GraphRAGFixture.GradeDocuments(input, config)

@pytest.mark.asyncio(loop_scope="function")
async def test_should_generate_answer(GraphRAGFixture):
    config = RunnableConfig(run_name=test_should_generate_answer.__name__, configurable={"thread_id": uuid7str()})
    input = {
        "messages": convert_to_messages(
            [
                {
                    "role": "user",
                    "content": "What does Lilian Weng say about types of reward hacking?",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "1",
                            "name": "retrieve_blog_posts",
                            "args": {"query": "types of reward hacking"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "reward hacking can be categorized into two types: environment or goal misspecification, and reward tampering",
                    "tool_call_id": "1",
                },
            ]
        )
    }
    assert "Generate" == await GraphRAGFixture.GradeDocuments(input, config)

@pytest.mark.asyncio(loop_scope="function")
async def test_rewrite_question(GraphRAGFixture):
    config = RunnableConfig(run_name=test_rewrite_question.__name__, configurable={"thread_id": uuid7str()})
    input = {
        "messages": convert_to_messages(
            [
                {
                    "role": "user",
                    "content": "What does Lilian Weng say about types of reward hacking?",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "1",
                            "name": "retrieve_blog_posts",
                            "args": {"query": "types of reward hacking"},
                        }
                    ],
                },
                {"role": "tool", "content": "meow", "tool_call_id": "1"},
            ]
        )
    }
    response = await GraphRAGFixture.Rewrite(input, config)
    print(f"{test_rewrite_question.__name__} response: {len(response["messages"][-1]["content"])}, {response["messages"][-1]["content"]}")
    assert len(response["messages"][-1]["content"])
   
@pytest.mark.asyncio(loop_scope="function")
async def test_generete_answer(GraphRAGFixture):
    config = RunnableConfig(run_name=test_generete_answer.__name__, configurable={"thread_id": uuid7str()})
    input = {
        "messages": convert_to_messages(
            [
                {
                    "role": "user",
                    "content": "What does Lilian Weng say about types of reward hacking?",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "1",
                            "name": "retrieve_blog_posts",
                            "args": {"query": "types of reward hacking"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "reward hacking can be categorized into two types: environment or goal misspecification, and reward tampering",
                    "tool_call_id": "1",
                },
            ]
        )
    }
    response = await GraphRAGFixture.Generate(input, config)
    lc_ai_message = response["messages"][-1]
    assert lc_ai_message
    ai_message = ChatMessage.from_langchain(lc_ai_message)
    print(f"{test_generete_answer.__name__} response: {len(ai_message.content)}, {response["messages"][-1].pretty_print()}")
    assert len(ai_message.content)
