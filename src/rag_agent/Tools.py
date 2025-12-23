"""This module provides example tools for web scraping and search functionality.
It includes a basic Tavily search function (as an example)
These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""
import asyncio, logging
from typing import Any, Callable, List, Optional, cast
from langgraph.runtime import Runtime
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig, ensure_config
from langchain_core.tools import InjectedToolArg, tool
from langgraph.store.base import BaseStore
from langgraph.prebuilt import InjectedStore
from langchain_core.documents import Document
from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from typing_extensions import Annotated
from uuid_extensions import uuid7, uuid7str
from src.common.configuration import Configuration
from src.config import config as appconfig
from src.common.State import CustomAgentState
from src.rag_agent.Context import Context
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
    model_id = "gemini-2.5-flash"
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
    return serialized, retrieved_docs

# https://langchain-ai.github.io/langgraph/concepts/memory/
# https://langchain-ai.github.io/langgraph/how-tos/cross-thread-persistence/
# https://github.com/langchain-ai/memory-agent
@tool(parse_docstring=True)
async def upsert_memory(
    content: str,
    context: str,
    *,
    memory_id: Optional[uuid7str] = None,
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg],
    store: Annotated[BaseStore, InjectedStore()],
):
    """Upsert a memory in the database.

    If a memory conflicts with an existing one, then just UPDATE the
    existing one by passing in memory_id - don't create two memories
    that are the same. If the user corrects a memory, UPDATE it.
    It uses a simple memory structure "content: str, context: str" for each memory, but it could be structured in other ways.

    Args:
        content: The main content of the memory. For example:
            "User expressed interest in learning about French."
        context: Additional context for the memory. For example:
            "This was mentioned while discussing career options in Europe."
        memory_id: ONLY PROVIDE IF UPDATING AN EXISTING MEMORY.
        The memory to overwrite.
    """
    logging.debug(f"upsert_memory content: {content}, context: {context}, memory_id: {memory_id}")
    mem_id = memory_id or uuid7str()
    user_id = Configuration.from_runnable_config(config).user_id
    logging.debug(f"upsert_memory user_id: {user_id}")
    await store.aput(
        ("memories", user_id),
        key=str(mem_id),
        value={"content": content, "context": context},
    )
    logging.debug(f"upsert_memory mem_id: {mem_id}")
    return f"Stored memory {mem_id}"

async def store_memory(state: CustomAgentState, runtime: Runtime[Context]):
    """
    Read from the agent/graph's `Store` to easily list extracted memories. 
    If it calls a tool, LangGraph will route to the `store_memory` node to save the information to the store.
    """
    # Extract tool calls from the last message
    tool_calls = getattr(state.messages[-1], "tool_calls", [])

    # Concurrently execute all upsert_memory calls
    saved_memories = await asyncio.gather(
        *(
            #upsert_memory(**tc["args"], user_id=runtime.context.user_id, config=config, store=store)
            upsert_memory(
                **tc["args"],
                user_id=runtime.context.user_id,
                store=cast(BaseStore, runtime.store),
            )
            for tc in tool_calls
        )
    )
    # Format the results of memory storage operations
    # This provides confirmation to the model that the actions it took were completed
    results = [
        {
            "role": "tool",
            "content": mem,
            "tool_call_id": tc["id"],
        }
        for tc, mem in zip(tool_calls, saved_memories)
    ]
    return {"messages": results}

if __name__ == "__main__":
    print(f"upsert_memory input schema: {upsert_memory.get_input_schema().model_json_schema()}, tool_call schema: ${upsert_memory.tool_call_schema.model_json_schema()}")