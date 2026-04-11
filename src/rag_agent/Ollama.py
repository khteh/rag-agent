import asyncio, logging, json
from datetime import datetime
from pathlib import Path
from uuid_extensions import uuid7, uuid7str
from asyncio import Queue, run, create_task
from langchain_ollama import ChatOllama
from langchain_core.tools import InjectedToolArg, tool
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig, ensure_config
from typing_extensions import List, TypedDict
from src.config import config
#llm = ChatOllama(model=config.LLM_RAG_MODEL, base_url=config.BASE_URI, api_key=config.OLLAMA_API_KEY, streaming=True, temperature=0, think="high")
llm = init_chat_model(config.LLM_RAG_MODEL, model_provider=config.MODEL_PROVIDER, base_url=config.BASE_URI, api_key=config.OLLAMA_API_KEY, streaming=True, temperature=0, think="high")

@tool
def echo(x: str) -> str:
    """
    Use this to echo the input string
    """
    return x

async def main():
    agent = llm.bind_tools([echo])
    response = await agent.ainvoke("Hello, wassup!?!")
    logging.debug(f"response: {response}")

if __name__ == "__main__":
    run(main())