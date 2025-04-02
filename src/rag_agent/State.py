"""Define the state structures for the agent."""
from __future__ import annotations
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from typing_extensions import List, TypedDict
from dataclasses import dataclass, field
from typing import Sequence
from langchain_core.messages import AnyMessage
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep
from langgraph.prebuilt.chat_agent_executor import AgentState
from typing_extensions import Annotated
from langgraph.graph.graph import (
    END,
    START,
    CompiledGraph,
    Graph,
    Send,
)
from src.schema.EmailModel import EmailModel
from pydantic import EmailStr

"""
Define state for application - The information each node in StateGraph updates and passes to the next node.
"""
@dataclass
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]    
    is_last_step: IsLastStep

@dataclass
class CustomAgentState(AgentState):
    context: List[Document]
    is_last_step: IsLastStep

@dataclass
class EmailRAGState(TypedDict):
    email: str
    extract: EmailModel | None
    escalation_text_criteria: str
    escalation_dollar_criteria: float
    escalate: bool
    escalation_emails: list[EmailStr] | None
    follow_ups: dict[str, bool] | None
    current_follow_up: str | None

@dataclass
class EmailAgentState(AgentState):
    context: EmailRAGState
    graph: CompiledGraph
