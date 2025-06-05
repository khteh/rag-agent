import locale
from datetime import datetime, date
from pydantic import BaseModel, Field, computed_field
class DocumentGradeModel(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )