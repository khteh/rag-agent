import atheris, asyncio, sys, logging
from datetime import datetime
from uuid_extensions import uuid7, uuid7str
from langchain_core.runnables import RunnableConfig, ensure_config
from src.rag_agent.RAGAgent import RAGAgent

async def main(input_message):
    rag = RAGAgent()
    await rag.CreateGraph()
    config = RunnableConfig(run_name="RAG Deep Agent", configurable={"thread_id": uuid7str(), "user_id": uuid7str()})
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    message = f"[Timestamp: {timestamp}]\n{input_message}\n"
    await rag.ChatAgent(config, message, ["values"], False)

def FuzzEntryPoint(data):
    # Initialize the provider with raw bytes from the fuzzer
    fdp = atheris.FuzzedDataProvider(data)    
    # Consume structured data
    number = fdp.ConsumeIntInRange(100, 100000)
    input_message = fdp.ConsumeUnicodeNoSurrogates(128)
    print(f"input_messsage: {repr(input_message)}", flush=True)
    asyncio.run(main(input_message))

if __name__ == "__main__":
    atheris.instrument_all()    
    atheris.Setup(sys.argv, FuzzEntryPoint)
    atheris.Fuzz()    
