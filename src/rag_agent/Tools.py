"""This module provides example tools for web scraping and search functionality.
It includes a basic Tavily search function (as an example)
These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""
from typing import Any, Callable, List, Optional, cast
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig, ensure_config
from langchain_core.tools import InjectedToolArg, tool
from langgraph.store.base import BaseStore
from langgraph.prebuilt import InjectedStore
from langchain.schema import Document
from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from typing_extensions import Annotated
from .configuration import Configuration
from src.config import config as appconfig

async def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    Needs Tavily API key
    """
    #configuration = Configuration.from_runnable_config(config)
    #wrapped = TavilySearchResults(max_results=configuration.max_search_results)
    config = ensure_config(config)
    max_search_results = config.get("configurable", {}).get("max_search_results")
    wrapped = TavilySearchResults(max_results=max_search_results)
    result = await wrapped.ainvoke({"query": query})
    return cast(list[dict[str, Any]], result)

"""
https://github.com/langchain-ai/langchain/discussions/30282
https://ai.google.dev/gemini-api/docs/text-generation?lang=python
@tool gives the function’s docstring to the agent’s LLM, helping it determine whether that particular tool is relevant to the task at hand.
"""
@tool
def ground_search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
)-> Optional[str]: #Optional[list[str]]
    """
    Search for general web results.
    
    This function performs a search using the Google search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    client = genai.Client(api_key=appconfig.GEMINI_API_KEY)
    model_id = "gemini-2.0-flash"
    google_search_tool = Tool(
        google_search = GoogleSearch()
    )
    # https://cloud.google.com/vertex-ai/docs/reference/rest/v1/GenerateContentResponse
    response = client.models.generate_content(
        model=model_id,
        contents=[{"role": "user", "parts": [{"text": query}]}], 
        config=GenerateContentConfig( #https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig
            tools=[google_search_tool],
            system_instruction="You are a helpful AI assistant named Bob.",
            response_modalities=["TEXT"], # https://ai.google.dev/api/generate-content#Modality
        )
    )
    # To get grounding metadata as web content. https://cloud.google.com/vertex-ai/docs/reference/rest/v1/GenerateContentResponse#GroundingMetadata
    #print(f"metadata: {response.candidates[0].grounding_metadata}")
    #print(response.candidates[0].grounding_metadata.search_entry_point.rendered_content)
    #return [content.text for content in response.candidates[0].content.parts]
    #return Document(page_content="\n\n".join(content.text for content in response.candidates[0].content.parts)) This doesn't work well. https://api.python.langchain.com/en/v0.0.339/schema/langchain.schema.document.Document.html
    return "\n\n".join(content.text for content in response.candidates[0].content.parts)

@tool(response_format="content_and_artifact")
async def retrieve(query: str, *, config: Annotated[RunnableConfig, InjectedToolArg], store: Annotated[BaseStore, InjectedStore()]):
    """Retrieve information related to a query."""
    retrieved_docs = await store.asimilarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs#

#https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent
@tool
async def save_memory(memory: str, *, config: Annotated[RunnableConfig, InjectedToolArg], store: Annotated[BaseStore, InjectedStore()]) -> str:
    """
    Save the given memory for the current user.
    This should only be used after you have accomplised your task and ready to respond to user request with an answer.
    """
    # This is a **tool** the model can use to save memories to storage
    config = ensure_config(config)
    user_id = config.get("configurable", {}).get("user_id")
    namespace = ("memories", user_id)
    store.put(namespace, f"memory_{len(await store.asearch(namespace))}", {"data": memory})
    return f"Saved memory: {memory}"

