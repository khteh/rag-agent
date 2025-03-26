import re, asyncio, json, logging
from uuid_extensions import uuid7
from typing import AsyncGenerator, Dict, Any, Tuple
from quart import Blueprint, render_template, session, current_app
from datetime import datetime, timezone
from quart import Quart
from werkzeug.exceptions import HTTPException
from contextlib import asynccontextmanager
from quart.helpers import stream_with_context
from src.schema.schema import ChatMessage, UserInput, StreamInput
from langchain_core.callbacks import AsyncCallbackHandler
from langgraph.graph.graph import CompiledGraph
from langchain_core.runnables import RunnableConfig
#from src.rag_agent.RAGAgent import agent
home_api = Blueprint("home", __name__)
@home_api.context_processor

class TokenQueueStreamingHandler(AsyncCallbackHandler):
    """LangChain callback handler for streaming LLM tokens to an asyncio queue."""
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        if token:
            await self.queue.put(token)

def _parse_input(user_input: UserInput) -> Tuple[Dict[str, Any], str]:
    run_id = uuid7()
    thread_id = user_input.thread_id or str(uuid7())
    input_message = ChatMessage(type="human", content=user_input.message)
    kwargs = dict(
        input={"messages": [input_message.to_langchain()]},
        config=RunnableConfig(
            configurable={"thread_id": thread_id},
            run_id=run_id,
        ),
    )
    return kwargs, run_id

@home_api.post("/invoke")
async def invoke(user_input: UserInput) -> ChatMessage:
    """
    Invoke the agent with user input to retrieve a final response.

    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    """
    kwargs, run_id = _parse_input(user_input)
    logging.debug(kwargs)
    try:
        response = await current_app.agent.ainvoke(**kwargs)
        output = ChatMessage.from_langchain(response["messages"][-1])
        output.run_id = str(run_id)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def message_generator(user_input: StreamInput) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint.
    """
    kwargs, run_id = _parse_input(user_input)

    # Use an asyncio queue to process both messages and tokens in
    # chronological order, so we can easily yield them to the client.
    output_queue = asyncio.Queue(maxsize=10)
    if user_input.stream_tokens:
        kwargs["config"]["callbacks"] = [TokenQueueStreamingHandler(queue=output_queue)]

    # Pass the agent's stream of messages to the queue in a separate task, so
    # we can yield the messages to the client in the main thread.
    async def run_agent_stream():
        async for s in current_app.agent.astream(**kwargs, stream_mode="updates"):
            await output_queue.put(s)
        await output_queue.put(None)

    stream_task = asyncio.create_task(run_agent_stream())

    # Process the queue and yield messages over the SSE stream.
    while s := await output_queue.get():
        if isinstance(s, str):
            # str is an LLM token
            yield f"data: {json.dumps({'type': 'token', 'content': s})}\n\n"
            continue

        # Otherwise, s should be a dict of state updates for each node in the graph.
        # s could have updates for multiple nodes, so check each for messages.
        new_messages = []
        for _, state in s.items():
            if "messages" in state:
                new_messages.extend(state["messages"])
        for message in new_messages:
            try:
                chat_message = ChatMessage.from_langchain(message)
                chat_message.run_id = str(run_id)
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': f'Error parsing message: {e}'})}\n\n"
                continue
            # LangGraph re-sends the input message, which feels weird, so drop it
            if chat_message.type == "human" and chat_message.content == user_input.message:
                continue
            yield f"data: {json.dumps({'type': 'message', 'content': chat_message.dict()})}\n\n"

    await stream_task
    yield "data: [DONE]\n\n"

@home_api.post("/stream")
async def stream_agent(user_input: StreamInput):
    """
    Stream the agent's response to a user input, including intermediate messages and tokens.

    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.
    """
    @stream_with_context
    async def async_generator():
        message = message_generator(user_input)
        yield message.encode()
    return async_generator(), 200
    #return StreamingResponse(message_generator(user_input), media_type="text/event-stream")