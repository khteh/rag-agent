"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""
import os
from dotenv import load_dotenv
from typing import Any, Callable, List, Optional, cast
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool
from langgraph.store.base import BaseStore
from langgraph.prebuilt import InjectedStore
from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from typing_extensions import Annotated
from configuration import Configuration
from VectorStore import vector_store
load_dotenv()
async def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    Needs Tavily API key
    """
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearchResults(max_results=configuration.max_search_results)
    result = await wrapped.ainvoke({"query": query})
    return cast(list[dict[str, Any]], result)

# https://github.com/langchain-ai/langchain/discussions/30282
@tool
async def GoogleSearch(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
)-> Optional[list[str]]:
    """Search for general web results.

    This function performs a search using the Google search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    model_id = "gemini-2.0-flash"
    google_search_tool = Tool(
        google_search = GoogleSearch()
    )
    response = await client.models.generate_content(
        model=model_id,
        contents=[{"role": "user", "parts": [{"text": query}]}], 
        config=GenerateContentConfig(
            tools=[google_search_tool],
            response_modalities=["TEXT"],
        )
    )
    result = []
    for each in response.candidates[0].content.parts:
        result.append(each.text)
    # Example response:
    # The next total solar eclipse visible in the contiguous United States will be on ...

    # To get grounding metadata as web content.
    #print(response.candidates[0].grounding_metadata.search_entry_point.rendered_content)
    return result

@tool(response_format="content_and_artifact")
async def retrieve(query: str, *, config: RunnableConfig):
    """Retrieve information related to a query."""
    retrieved_docs = await vector_store.asimilarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

@tool
async def save_memory(memory: str, *, config: RunnableConfig, store: Annotated[BaseStore, InjectedStore()]) -> str:
    '''Save the given memory for the current user.'''
    # This is a **tool** the model can use to save memories to storage
    user_id = config.get("configurable", {}).get("user_id")
    namespace = ("memories", user_id)
    store.put(namespace, f"memory_{len(await store.asearch(namespace))}", {"data": memory})
    return f"Saved memory: {memory}"

TOOLS: List[Callable[..., Any]] = [retrieve, GoogleSearch, save_memory]
