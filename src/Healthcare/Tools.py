from langchain_core.tools import InjectedToolArg, tool, Tool
from typing_extensions import Annotated
from langchain_core.runnables import RunnableConfig, ensure_config
from langchain_core.tools import InjectedToolArg, tool
from langgraph.store.base import BaseStore
from langgraph.prebuilt import InjectedStore
from .HospitalWaitingTime import get_current_wait_times, get_most_available_hospital
from .HospitalReviewChain import reviews_vector_chain, hospital_cypher_chain
from src.rag_agent.Tools import save_memory

TOOLS = [
    Tool(
        name = "Experiences",
        func = reviews_vector_chain.ainvoke,
        description="""Useful when you need to answer questions
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
        func=hospital_cypher_chain.ainvoke,
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
    Tool(
        name="SaveMemory",
        func=save_memory,
        description="""Save the given memory for the current user."""
    )
]