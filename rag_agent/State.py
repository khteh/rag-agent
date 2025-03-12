from langchain_core.documents import Document
from typing_extensions import List, TypedDict
# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
