import locale
from datetime import datetime, date
from pydantic import BaseModel, Field, computed_field

class EmailModel(BaseModel):
    date_str: str | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="The date of the email reformatted to match mm-dd-YYYY. This is usually found in the Date: field in the email. Ignore the timestamp and timezone part of the Date:",
    )
    name: str | None = Field(
        default=None,
        description="The name of the email sender. This is usually found in the From: field in the email formatted as name <email>",
    )
    phone: str | None = Field(
        default=None,
        description="The phone number of the email sender (if present in the message). This is usually found in the signature at the end of the email body.",
    )
    email: str | None = Field(
        default=None,
        description="The email addreess of the email sender (if present in the message). This is usually found in the From: field in the email formatted as name <email>",
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
        """
        locale.getlocale()
        local.setlocale(local.LC_ALL, 'en_US')
        https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
        Date: Wed, 02 Apr 2025 15:39:59 -0700
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