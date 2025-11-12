"""Define the state structures for the agent."""
from __future__ import annotations
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from dataclasses import dataclass, field
from typing import Sequence
from langchain_core.messages import AnyMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep
from langchain.agents import AgentState
from typing_extensions import Annotated
from langgraph.graph.state import CompiledStateGraph
from src.models.EmailModel import EmailModel
from pydantic import EmailStr

"""
Define state for application - The information each node in StateGraph updates and passes to the next node.
https://python.langchain.com/docs/tutorials/rag/
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

# https://github.com/langchain-ai/langgraph/blob/62b2580ad5101cf55da0b6bebcd09913d2512022/libs/prebuilt/langgraph/prebuilt/chat_agent_executor.py#L57
@dataclass
class CustomAgentState(AgentState):
    context: List[Document]

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
    graph: CompiledStateGraph
