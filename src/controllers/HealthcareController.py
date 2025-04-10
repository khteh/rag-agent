import re, asyncio, json, logging
from uuid_extensions import uuid7
from typing import AsyncGenerator, Dict, Any, Tuple
from quart import Blueprint, render_template, session, current_app
from datetime import datetime, timezone
from quart import (
    Blueprint,
    request,
    Response,
    ResponseReturnValue,
    current_app,
    make_response,
    render_template,
    session
)
from urllib.parse import urlparse, parse_qs
from werkzeug.exceptions import HTTPException
from contextlib import asynccontextmanager
from quart.helpers import stream_with_context
from src.models.schema import ChatMessage, UserInput, StreamInput
from langchain_core.callbacks import AsyncCallbackHandler
from langgraph.graph.graph import CompiledGraph
from langchain_core.runnables import RunnableConfig
from src.utils.AsyncRetry import async_retry
from src.common.Response import custom_response
from src.models.schema import UserInput
from src.models.HealthcareModel import HospitalQueryOutput
from src.utils.InputParser import parse_input
healthcare_api = Blueprint("healthcare", __name__)
@healthcare_api.context_processor

@async_retry(max_retries=10, delay=1)
async def invoke_agent_with_retry(query: str):
    """Retry the agent if a tool fails to run.

    This can help when there are intermittent connection issues
    to external APIs.
    """
    return await current_app.healthcare_agent.ainvoke({"input": query})

@healthcare_api.post("/rag")
async def query_hospital_agent(query: UserInput) -> HospitalQueryOutput:
    query_response = await invoke_agent_with_retry(query.message)
    query_response["intermediate_steps"] = [
        str(s) for s in query_response["intermediate_steps"]
    ]
    return query_response

@healthcare_api.post("/invoke")
async def invoke(): #user_input: UserInput) -> ChatMessage:
    """
    Invoke the agent with user input to retrieve a final response.

    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    """
    data = await request.get_data()
    params = parse_qs(data.decode('utf-8'))
    logging.debug(f"data: {data}, params: {params}")
    user_input: UserInput = json.loads(data)
    kwargs, run_id = parse_input(user_input)
    logging.debug(kwargs)
    try:
        response = await current_app.healthcare_agent.ainvoke(**kwargs)
        output = ChatMessage.from_langchain(response["messages"][-1])
        output.run_id = str(run_id)
        #return output
        #return await Respond("index.html", title="Welcome to Python LLM-RAG", greeting=greeting)
        # res.json({ 'message': this.presenter.Message, "errors": this.presenter.Errors });
        logging.debug(f"/invoke respose: {output}")
        return custom_response(output, 200)
    except Exception as e:
        raise HTTPException(description = str(e))
