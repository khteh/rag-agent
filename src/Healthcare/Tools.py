from langchain_core.tools import InjectedToolArg, tool, Tool
from typing_extensions import Annotated
from langchain_core.runnables import RunnableConfig, ensure_config
from langgraph.store.base import BaseStore
from langgraph.prebuilt import InjectedStore
from langchain.prompts import PromptTemplate
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from src.rag_agent.Tools import save_memory
from .HospitalWaitingTime import get_current_wait_times, get_most_available_hospital
from .prompts import cypher_generation_template, qa_generation_template
from .HospitalReviewChain import reviews_vector_chain
from .HospitalCypherChain import hospital_cypher_chain

TOOLS = [
    Tool(
        name = "Experiences",
        coroutine = reviews_vector_chain.ainvoke,
        func = reviews_vector_chain.invoke,
        description ="""Useful when you need to answer questions
        about patient experiences, feelings, or any other qualitative
        question that could be answered about a patient using semantic
        search. Not useful for answering objective questions that involve
        counting, percentages, aggregations, or listing facts. Use the
        entire prompt as input to the tool. For instance, if the prompt is
        "Are patients satisfied with their care?", the input should be
        "Are patients satisfied with their care?".
        """,
    ),
    Tool(
        name="Graph",
        coroutine = hospital_cypher_chain.ainvoke,
        func = hospital_cypher_chain.invoke,
        description="""Useful for answering questions about patients,
        physicians, hospitals, insurance payers, patient review
        statistics, and hospital visit details. Use the entire prompt as
        input to the tool. For instance, if the prompt is "How many visits
        have there been?", the input should be "How many visits have
        there been?".
        """,
    ),
    Tool(
        name="WaitTimes",
        func=get_current_wait_times,
        description="""Use when asked about current wait times
        at a specific hospital. This tool can only get the current
        wait time at a hospital and does not have any information about
        aggregate or historical wait times. Do not pass the word "hospital"
        as input, only the hospital name itself. For example, if the prompt
        is "What is the current wait time at Jordan Inc Hospital?", the
        input should be "Jordan Inc".
        """,
    ),
    Tool(
        name="Availability",
        func=get_most_available_hospital,
        description="""
        Use when you need to find out which hospital has the shortest
        wait time. This tool does not have any information about aggregate
        or historical wait times. This tool returns a dictionary with the
        hospital name as the key and the wait time in minutes as the value.
        """,
    ),
    save_memory
]

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