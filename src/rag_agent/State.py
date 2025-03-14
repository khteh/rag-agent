"""Define the state structures for the agent."""
from __future__ import annotations
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from typing_extensions import List, TypedDict
from dataclasses import dataclass, field
from typing import Sequence
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from langgraph.prebuilt.chat_agent_executor import AgentState
from typing_extensions import Annotated

# Define state for application
@dataclass
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    is_last_step: IsLastStep

@dataclass
class CustomAgentState(AgentState):
    context: List[Document]
    is_last_step: IsLastStep
