"""Define the immutable context data for the agent."""
from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Annotated, Optional
from langchain_core.runnables import RunnableConfig, ensure_config
from langchain_core.prompts import PromptTemplate
from langgraph.graph.state import CompiledStateGraph
from .Prompts import SYSTEM_PROMPT
from src.Healthcare.prompts import cypher_generation_template, qa_generation_template, review_template
from .State import EmailRAGState
from src.Infrastructure.VectorStore import VectorStore
@dataclass(kw_only=True)
class ContextSchema:
    """The configuration for the agent."""
    user_id: str = "default"
    thread_id:str = None
