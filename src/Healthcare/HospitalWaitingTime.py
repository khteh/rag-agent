import logging, numpy
from typing_extensions import Annotated
from langchain_core.runnables import RunnableConfig, ensure_config
from langchain_core.tools import InjectedToolArg, tool, Tool
from langchain_neo4j import Neo4jGraph
from src.config import config
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())

def _get_current_hospitals() -> list[str]:
    """Fetch a list of current hospital names from a Neo4j database."""
    graph = Neo4jGraph(
        url=config.NEO4J_URI,
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
    )
    current_hospitals = graph.query(
        """
        MATCH (h:Hospital)
        RETURN h.name AS hospital_name
        """
    )
    return [d["hospital_name"].lower() for d in current_hospitals]

def _get_current_wait_time_minutes(hospital: str) -> int:
    """Get the current wait time at a hospital in minutes."""
    current_hospitals = _get_current_hospitals()
    if hospital.lower() not in current_hospitals:
        return -1
    return rng.integers(low=0, high=600, size=1) # random integer between 0 and 600 simulating a wait time in minutes.

@tool(description="""Use when asked about current wait times
        at a specific hospital. This tool can only get the current
        wait time at a hospital and does not have any information about
        aggregate or historical wait times. Do not pass the word "hospital"
        as input, only the hospital name itself. For example, if the prompt
        is "What is the current wait time at Jordan Inc Hospital?", the
        input should be "Jordan Inc".
        """)
def get_current_wait_times(query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]) -> str:
    """Get the current wait time at a hospital formatted as a string."""
    logging.info(f"\n=== get_current_wait_times ===")
    wait_time_in_minutes = _get_current_wait_time_minutes(query)
    if wait_time_in_minutes == -1:
        return f"Hospital '{query}' does not exist."
    hours, minutes = divmod(wait_time_in_minutes, 60)
    if hours > 0:
        return f"{hours} hours {minutes} minutes"
    else:
        return f"{minutes} minutes"

@tool(description="""
        Use when you need to find out which hospital has the shortest
        wait time. This tool does not have any information about aggregate
        or historical wait times. This tool returns a dictionary with the
        hospital name as the key and the wait time in minutes as the value.
        """)
def get_most_available_hospital(query: str = "", *, config: Annotated[RunnableConfig, InjectedToolArg]) -> dict[str, float]:
    """Find the hospital with the shortest wait time."""
    logging.info(f"\n=== get_most_available_hospital ===")
    current_hospitals = _get_current_hospitals()
    current_wait_times = [
        _get_current_wait_time_minutes(h) for h in current_hospitals
    ]
    best_time_idx = numpy.argmin(current_wait_times)
    best_hospital = current_hospitals[best_time_idx]
    best_wait_time = current_wait_times[best_time_idx]
    logging.debug(f"{best_hospital}: {best_wait_time}")
    return {best_hospital: best_wait_time}