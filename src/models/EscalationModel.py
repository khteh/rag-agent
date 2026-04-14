import re
from datetime import datetime, date
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator
from pydantic.fields import AliasChoices

class EscalationCheckModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    needs_escalation: bool = Field(
        description="Whether the email requires escalation according to the specified criteria",
        # Trace (Apr 10, khteh): LLM returned "immediate_escalation" instead of "needs_escalation"
        validation_alias=AliasChoices(
            "needs_escalation",
            "immediate_escalation",
            "requires_escalation",
            "escalation_required",
        ),        
    )
    escalation_reason: str | None = Field(
        default=None,
        description=(
            "Concise explanation of why escalation is (or is not) required, "
            "referencing the specific criteria met"
        ),
        # Trace (Apr 10, khteh): LLM returned "reason" instead of "escalation_reason"
        validation_alias=AliasChoices("escalation_reason", "reason", "explanation"),        
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
        # Trace (Apr 10, khteh): LLM returned "priority" instead of "escalation_priority"
        validation_alias=AliasChoices("escalation_priority", "priority"),        
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
            yes_no = re.search(
                r"(?:escalation\s+required|immediate_escalation)[^\w]*\*+\s*:?\s*\*+\s*(yes|no|true|false)",
                data,
                re.I,
            )
            reason = re.search(
                r"(?:escalation_reason|reason)\*+\s*:?\s*\**(.*?)(?:\n\n|\Z)",
                data,
                re.I | re.S,
            )
            priority = re.search(
                r"(?:escalation_priority|priority)\*+\s*:?\s*[`*]*(immediate|urgent|standard)[`*]*",
                data,
                re.I,
            )
            return {
                "needs_escalation": yes_no.group(1) if yes_no else "false",
                "escalation_reason": reason.group(1).strip() if reason else data.strip(),
                "escalation_priority": priority.group(1).lower() if priority else None,
            }
        if isinstance(data, dict):
            # Belt-and-suspenders: remap wrong dict key names the LLM may produce
            # even when AliasChoices cannot catch them (e.g. after JSON re-serialisation).
            # Trace (Apr 10, khteh): immediate_escalation, reason, priority
            key_remap = {
                "immediate_escalation": "needs_escalation",
                "requires_escalation": "needs_escalation",
                "escalation_required": "needs_escalation",
                "reason": "escalation_reason",
                "explanation": "escalation_reason",
                "priority": "escalation_priority",
            }
            for old, new in key_remap.items():
                if old in data and new not in data:
                    data[new] = data.pop(old)
            if "needs_escalation" not in data:
                # A present escalation_priority always implies escalation is needed.
                data["needs_escalation"] = data.get("escalation_priority") is not None
        return data