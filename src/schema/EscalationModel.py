from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class EscalationCheckModel(BaseModel):
    needs_escalation: bool = Field(
        description="""Whether the notice requires escalation
        according to specified criteria"""
    )