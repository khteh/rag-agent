import os
from typing import Any
import numpy as np
from langchain_core.tools import InjectedToolArg, tool, Tool
from langchain_neo4j import Neo4jGraph
from src.config import config

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
    return np.random.randint(low=0, high=600) # random integer between 0 and 600 simulating a wait time in minutes.

@tool(description="""Use when asked about current wait times
        at a specific hospital. This tool can only get the current
        wait time at a hospital and does not have any information about
        aggregate or historical wait times. Do not pass the word "hospital"
        as input, only the hospital name itself. For example, if the prompt
        is "What is the current wait time at Jordan Inc Hospital?", the
        input should be "Jordan Inc".
        """)
def get_current_wait_times(hospital: str) -> str:
    """Get the current wait time at a hospital formatted as a string."""
    wait_time_in_minutes = _get_current_wait_time_minutes(hospital)
    if wait_time_in_minutes == -1:
        return f"Hospital '{hospital}' does not exist."
    hours, minutes = divmod(wait_time_in_minutes, 60)
    if hours > 0:
        return f"{hours} hours {minutes} minutes"
    else:
        return f"{minutes} minutes"
    
def get_most_available_hospital(_: Any) -> dict[str, float]:
    """Find the hospital with the shortest wait time."""
    current_hospitals = _get_current_hospitals()
    current_wait_times = [
        _get_current_wait_time_minutes(h) for h in current_hospitals
    ]
    best_time_idx = np.argmin(current_wait_times)
    best_hospital = current_hospitals[best_time_idx]
    best_wait_time = current_wait_times[best_time_idx]
    return {best_hospital: best_wait_time}