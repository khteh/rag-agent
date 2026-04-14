import locale
from datetime import datetime, date
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator
from pydantic.fields import AliasChoices

class EmailModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    date_str: str | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="The date of the email reformatted to match dd-mm-YYYY. This is usually found in the Date: field in the email. Ignore the timestamp and timezone part of the Date:",
        # Trace (Apr 10): LLM returned key "date" instead of "date_str"
        validation_alias=AliasChoices("date_str", "date"),        
    )
    name: str | None = Field(
        default=None,
        description="The name of the email sender. This is usually found in the From: field in the email formatted as name <email>",
        # Trace (Apr 8): LLM returned key "from_name" instead of "name"
        validation_alias=AliasChoices("name", "from_name"),        
    )
    phone: str | None = Field(
        default=None,
        description="The phone number of the email sender (if present in the message). This is usually found in the signature at the end of the email body.",
        # Trace (Apr 8+10): LLM returned key "phone_number" instead of "phone"
        validation_alias=AliasChoices("phone", "phone_number"),        
    )
    email: str | None = Field(
        default=None,
        description="The email addreess of the email sender (if present in the message). This is usually found in the From: field in the email formatted as name <email>",
        # Trace (Apr 8): LLM returned key "from_email" instead of "email"
        validation_alias=AliasChoices("email", "from_email"),        
    )
    project_id: int | None = Field(
        default=None,
        description="The project ID (if present in the message) - must be an integer. This is usually found in the Subject: field or email body text",
    )
    site_location: str | None = Field(
        default=None,
        description="The site location of the project (if present in the message). Use the full address if possible.",
    )
    violation_types: list[str] | None = Field(
        default=None,
        description="The types of violation (if present in the message)",
        # Trace (Apr 10): LLM returned key "violation_type" (singular) instead of "violation_types"
        validation_alias=AliasChoices("violation_types", "violation_type"),        
    )
    required_changes: list[str] | None = Field(
        default=None,
        description="The required changes specified by the email (if present in the message)",
    )
    compliance_deadline_str: str | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="The date that the company must comply (if any) reformatted to match dd-mm-YYYY",
        # Trace (Apr 8+10): LLM returned key "compliance_deadline" instead of "compliance_deadline_str"
        validation_alias=AliasChoices("compliance_deadline_str", "compliance_deadline"),        
    )
    max_potential_fine: float | None = Field(
        default=None,
        description="The maximum potential fine (if any) - must be an float",
        # Trace (Apr 10): LLM returned key "maximum_potential_fine" instead of "max_potential_fine"
        validation_alias=AliasChoices("max_potential_fine", "maximum_potential_fine"),        
    )

    @model_validator(mode="before")
    @classmethod
    def parse_markdown_table_fallback(cls, data: object) -> object:
        if not isinstance(data, str):
            return data
        # Trace (Apr 13): gpt-oss returned a markdown table instead of JSON
        # e.g. "| **date** | 03-04-2026 |"
        result: dict = {}
        key_map = {
            "date": "date_str",
            "date_str": "date_str",
            "from_name": "name",
            "name": "name",
            "from_email": "email",
            "email": "email",
            "phone_number": "phone",
            "phone": "phone",
            "project_id": "project_id",
            "site_location": "site_location",
            "violation_types": "violation_types",
            "violation_type": "violation_types",
            "required_changes": "required_changes",
            "compliance_deadline": "compliance_deadline_str",
            "compliance_deadline_str": "compliance_deadline_str",
            "max_potential_fine": "max_potential_fine",
            "maximum_potential_fine": "max_potential_fine",
        }
        for line in data.splitlines():
            # match "| key | value |" markdown table rows
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) < 2:
                continue
            raw_key = re.sub(r"\*+", "", parts[0]).strip().lower().replace(" ", "_")
            raw_val = re.sub(r"\*+|<br>", " ", parts[1]).strip()
            if not raw_key or not raw_val or raw_val in ("-", "—", "N/A", "null", ""):
                continue
            canonical = key_map.get(raw_key)
            if canonical:
                result[canonical] = raw_val
        return result if result else data
    
    @staticmethod
    def _convert_string_to_date(date_str: str | None) -> date | None:
        """
        locale.getlocale()
        local.setlocale(local.LC_ALL, 'en_US')
        https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
        Date: Wed, 02 Apr 2026 15:39:59 -0700
        """
        try:
            #return datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %z') if date_str else None
            return datetime.strptime(date_str, "%d-%m-%Y").date() if date_str else None
        except Exception as e:
            print(e)
            return None

    @computed_field
    @property
    def date_of_email(self) -> date | None:
        return self._convert_string_to_date(self.date_str) if self.date_str else None

    @computed_field
    @property
    def compliance_deadline(self) -> date | None:
        return self._convert_string_to_date(self.compliance_deadline_str) if self.compliance_deadline_str else None
    
    @field_validator("violation_types", "required_changes", mode="before")
    @classmethod
    def coerce_to_list(cls, v: object) -> list[str] | None:
        if v is None:
            return None
        if isinstance(v, str):
            # Trace (Apr 10): LLM returned semicolon-joined string e.g.
            # "Electrical Wiring; Fire Safety; Structural Integrity"
            if ";" in v:
                return [item.strip() for item in v.split(";") if item.strip()]
            # Trace (Apr 13): markdown table cell used "- item  - item2" (space-separated)
            if " - " in v or v.startswith("- "):
                items = [item.lstrip("- ").strip() for item in re.split(r"\s*-\s+", v) if item.strip()]
                if items:
                    return items
            return [v]
        return v  # type: ignore[return-value]