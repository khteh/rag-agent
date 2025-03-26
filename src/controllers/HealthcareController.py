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
from src.utils.AsyncRetry import async_retry
from src.schema.schema import UserInput
from src.schema.HealthcareModel import HospitalQueryOutput
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