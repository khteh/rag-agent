"""Define the configurable parameters for the agent."""
from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Annotated, Optional
from langchain_core.runnables import RunnableConfig, ensure_config
from langgraph.graph.graph import (
    END,
    START,
    CompiledGraph,
    Graph,
    Send,
)
from .prompts import SYSTEM_PROMPT
from .State import EmailRAGState
from .VectorStore import VectorStore
@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""
    user_id: str = "default"
    system_prompt: str = field(
        default=SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="google_genai/gemini-2.0-flash",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )
    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )
    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        config = ensure_config(config)
        _configurable = config.get("configurable", {})
        _fields = {f.name for f in fields(cls) if f.init}
         # Using a double asterisk before the argument will allow you to pass a variable number of keyword parameters in the function
        return cls(**{k: v for k, v in _configurable.items() if k in _fields})

@dataclass(kw_only=True)
class EmailConfiguration(Configuration):
    """The configuration for the email agent."""
    graph: CompiledGraph = None
    email_state: EmailRAGState = None