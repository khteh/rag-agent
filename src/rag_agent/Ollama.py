import asyncio, logging, json
from asyncio import Queue, run, create_task
from langchain_ollama import ChatOllama
from langchain_core.tools import InjectedToolArg, tool
from src.config import config
llm = ChatOllama(model=config.LLM_RAG_MODEL, base_url=config.BASE_URI, streaming=True, temperature=0)

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