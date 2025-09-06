import asyncio, logging, os, vertexai
from uuid_extensions import uuid7, uuid7str
from typing import Annotated, Literal, Sequence
from datetime import datetime
from google.api_core.exceptions import ResourceExhausted
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, MessagesState
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache
from langgraph.graph.graph import (
    END,
    START,
)
from langgraph.graph import CompiledStateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg_pool import AsyncConnectionPool, ConnectionPool
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate, SystemMessagePromptTemplate
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import List, TypedDict
from langgraph.store.memory import InMemoryStore
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langchain_core.tools import InjectedToolArg, tool
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent, InjectedStore
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel, Field
"""
https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/
https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
https://python.langchain.com/docs/tutorials/qa_chat_history/
https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/message.py
https://langchain-ai.github.io/langgraph/how-tos/streaming/#values
https://python.langchain.com/docs/how_to/configure/
"""
from src.config import config as appconfig
from src.common.State import EmailRAGState, EmailAgentState
from src.utils.image import show_graph
from src.models import ChatMessage
from src.models.EmailModel import EmailModel
from src.models.EscalationModel import EscalationCheckModel
from src.Infrastructure.VectorStore import VectorStore
from src.Infrastructure.PostgreSQLSetup import PostgreSQLCheckpointerSetup, PostgreSQLStoreSetup
from data.sample_emails import EMAILS
from src.common.configuration import EmailConfiguration

email_sub_agent_prompt = """You are an expert email parser.
Extract date from the Date: field, name and email from the From: field, project id from the Subject: field or email body text,
phone number, site location, violation type, required changes, compliance deadline, and maximum potential fine from the email body text.
If any of the fields aren't present, don't populate them. Don't populate fields if they're not present in the email.
Try to cast dates into the dd-mm-YYYY format. Ignore the timestamp and timezone part of the Date. 
Determine whether the following email received from a regulatory body requires immediate escalation.
Immediate escalation is required when {escalation_criteria}.

Here's the email:
{email}
"""
chainLLM = init_chat_model(appconfig.LLM_RAG_MODEL, model_provider="ollama", base_url=appconfig.OLLAMA_URI, temperature=0)
email_parser_chain = (
    email_parser_prompt
    | chainLLM.with_structured_output(EmailModel)
)
escalation_chain = (
    escalation_prompt
    | chainLLM.with_structured_output(EscalationCheckModel)
)

cache_policy = CachePolicy(ttl=600) # 10 minutes
graph_builder = StateGraph(EmailRAGState)
graph_builder.add_node("ParseEmail", self.ParseEmail, cache_policy = cache_policy)
graph_builder.add_node("NeedsEscalation", self.NeedsEscalation, cache_policy = cache_policy)
graph_builder.add_edge(START, "ParseEmail")
graph_builder.add_edge("ParseEmail", "NeedsEscalation")
graph_builder.add_edge("NeedsEscalation", END)
graph = graph_builder.compile(name="Email RAG StateGraph", cache=InMemoryCache())

@tool
async def email_processing_tool(
    email: str, escalation_criteria: str,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> EmailModel:
    """
    Extract structured fields from a regulatory email.
    This should be used when the email message comes from
    a regulatory body or auditor regarding a property or
    construction site that the company works on.

    escalation_criteria is a description of which kinds of
    notices require immediate escalation.

    After calling this tool, you don't need to call any others.
    """
    logging.info(f"\n=== email_processing_tool ===")
    """Extract the user's state from the conversation and update the memory."""
    #graph = EmailConfiguration.from_runnable_config(config).graph
    emailState = EmailConfiguration.from_runnable_config(config).email_state
    emailState["email"] = email
    emailState["escalation_text_criteria"] = escalation_criteria
    logging.debug(f"email: {email}, escalation_criteria: {escalation_criteria}, emailState:: {emailState}")
    results = await graph.with_config(config).ainvoke(emailState)
    logging.debug(f"result: {results}")
    return results["extract"]

email_sub_agent = {
    "name": "email-rag-agent",
    "description": "Used to extract useful information from email given as input and determine if the email needs escalation based on the escalation criteria provided as input.",
    "prompt": email_sub_agent_prompt,
    "tools": [email_processing_tool],
}