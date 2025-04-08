from pydantic import BaseModel, Field

class EscalationCheckModel(BaseModel):
    needs_escalation: bool = Field(
        description="Whether the email requires escalation according to specified criteria"
    )