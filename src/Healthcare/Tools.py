from langchain_core.tools import InjectedToolArg, tool, Tool
from typing_extensions import Annotated
from langchain_core.runnables import RunnableConfig, ensure_config
from .HospitalReviewChain import reviews_vector_chain
from .HospitalCypherChain import hospital_cypher_chain

@tool(description ="""Useful when you need to answer questions
        about patient experiences, feelings, or any other qualitative
        question that could be answered about a patient using semantic
        search. Not useful for answering objective questions that involve
        counting, percentages, aggregations, or listing facts. Use the
        entire prompt as input to the tool. For instance, if the prompt is
        "Are patients satisfied with their care?", the input should be
        "Are patients satisfied with their care?".
        """)
async def HealthcareReview(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> dict[str, float]:
    return await reviews_vector_chain.ainvoke(query)

@tool(description="""Useful for answering questions about patients,
        physicians, hospitals, insurance payers, patient review
        statistics, and hospital visit details. Use the entire prompt as
        input to the tool. For instance, if the prompt is "How many visits
        have there been?", the input should be "How many visits have
        there been?""")
async def HealthcareCypher(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> dict[str, float]:
    return await hospital_cypher_chain.ainvoke(query)