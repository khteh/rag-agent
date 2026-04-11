import re
from pydantic import BaseModel, Field, field_validator, model_validator

class EscalationCheckModel(BaseModel):
    needs_escalation: bool = Field(
        description="Whether the email requires escalation according to the specified criteria",
    )
    escalation_reason: str | None = Field(
        default=None,
        description=(
            "Concise explanation of why escalation is (or is not) required, "
            "referencing the specific criteria met"
        ),
    )
    escalation_priority: str | None = Field(
        default=None,
        description=(
            "Priority level when escalation is needed: "
            "'immediate' (safety risk or stop-work order threat), "
            "'urgent' (large potential fine or tight compliance deadline), "
            "or 'standard' (other criteria met). "
            "Leave null when needs_escalation is false."
        ),
    )

    @field_validator("needs_escalation", mode="before")
    @classmethod
    def coerce_bool(cls, v: object) -> bool:
        if isinstance(v, str):
            return v.strip().lower() in ("yes", "true", "1")
        return bool(v)

    @model_validator(mode="before")
    @classmethod
    def parse_markdown_fallback(cls, data: object) -> object:
        if isinstance(data, str):
            # The model returned markdown instead of JSON — extract fields with regex.
            yes_no = re.search(r"escalation\s+required[^\w]*\*+\s*:?\s*\*+\s*(yes|no|true|false)", data, re.I)
            reason = re.search(r"escalation_reason\*+\s*:?\s*\**(.*?)(?:\n\n|\Z)", data, re.I | re.S)
            priority = re.search(r"escalation_priority\*+\s*:?\s*[`*]*(immediate|urgent|standard)[`*]*", data, re.I)
            return {
                "needs_escalation": yes_no.group(1) if yes_no else "false",
                "escalation_reason": reason.group(1).strip() if reason else data.strip(),
                "escalation_priority": priority.group(1).lower() if priority else None,
            }
        if isinstance(data, dict) and "needs_escalation" not in data:
            # The model omitted needs_escalation but included enough context to infer it.
            # A present escalation_priority always implies escalation is needed.
            data["needs_escalation"] = data.get("escalation_priority") is not None
        return data