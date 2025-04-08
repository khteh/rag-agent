from pydantic import BaseModel

class HospitalQueryOutput(BaseModel):
    """
    Response body
    """
    input: str
    output: str
    intermediate_steps: list[str]