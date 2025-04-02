from datetime import datetime, date
from pydantic import BaseModel, Field, computed_field

class EmailModel(BaseModel):
    """
    Example:
    Date: December 25, 2024
    From: Mickey Mouse <mickey@mouse.com>
    """
    date_str: str | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="The date of the email reformatted to match YYYY-mm-dd. This is usually found in the \"Date:\" field in the email.",
    )
    name: str | None = Field(
        default=None,
        description="The name of the email sender. This is usually found in the \"From:\" field in the email formatted as \"name <email>\"",
    )
    phone: str | None = Field(
        default=None,
        description="The phone number of the email sender (if present in the message). This is usually found in the signature at the end of the email body.",
    )
    email: str | None = Field(
        default=None,
        description="The email addreess of the email sender (if present in the message). This is usually found in the same \"From:\" field in the email formatted as \"name <email>\"",
    )
    project_id: int | None = Field(
        default=None,
        description="The project ID (if present in the message) - must be an integer",
    )
    site_location: str | None = Field(
        default=None,
        description="The site location of the project (if present in the message). Use the full address if possible.",
    )
    violation_type: str | None = Field(
        default=None,
        description="The type of violation (if present in the message)",
    )
    required_changes: str | None = Field(
        default=None,
        description="The required changes specified by the email (if present in the message)",
    )
    compliance_deadline_str: str | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="The date that the company must comply (if any) reformatted to match YYYY-mm-dd",
    )
    max_potential_fine: float | None = Field(
        default=None,
        description="The maximum potential fine (if any)",
    )

    @staticmethod
    def _convert_string_to_date(date_str: str | None) -> date | None:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").date() if date_str else None
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