"""Define the immutable context data for the agent."""
from __future__ import annotations
from dataclasses import dataclass, field, fields
@dataclass(kw_only=True)
class ContextSchema:
    """The configuration for the agent."""
    user_id: str = "default"
    thread_id:str = None
