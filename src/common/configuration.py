"""Define the configurable parameters for the agent."""
from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Annotated, Optional
from langchain_core.runnables import RunnableConfig, ensure_config
from langchain.prompts import PromptTemplate
from langgraph.graph.state import CompiledStateGraph
from .prompts import SYSTEM_PROMPT
from src.Healthcare.prompts import cypher_generation_template, qa_generation_template, review_template
from .State import EmailRAGState
from src.Infrastructure.VectorStore import VectorStore
@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""
    user_id: str = "default"
    """The ID of the user to remember in the conversation."""
    system_prompt: str = field(
        default = SYSTEM_PROMPT,
        metadata = {
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )
    cypher_generation_prompt: str = field(
        default = cypher_generation_template,
        metadata = {
            "description": "The prompt to generate Cypher query for a Neo4j graph database."
        }
    )
    qa_generation_prompt: str = field(
        default = qa_generation_template,
        metadata = {
            "description": "The prompt to take the results from a Neo4j Cypher query and forms a human-readable response."
        }
    )
    review_prompt: str = field(
        default = review_template,
        metadata = {
            "description": "the prompt to use patient reviews to answer questions about their experience at a hospital."
        }
    )
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default= "ollama/llama3.3", #"google_genai/gemini-2.0-flash",
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
        configurable = config.get("configurable", {})
        _fields = {f.name for f in fields(cls) if f.init}
         # Using a double asterisk before the argument will allow you to pass a variable number of keyword parameters in the function
        return cls(**{k: v for k, v in configurable.items() if k in _fields})

@dataclass(kw_only=True)
class EmailConfiguration(Configuration):
    """The configuration for the email agent."""
    graph: CompiledStateGraph = None
    email_state: EmailRAGState = None