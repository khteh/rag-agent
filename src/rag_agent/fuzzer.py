import atheris, asyncio, sys
from uuid_extensions import uuid7, uuid7str
from langchain_core.runnables import RunnableConfig, ensure_config
from src.models.schema import ChatMessage, UserInput, StreamInput
from src.rag_agent.RAGAgent import RAGAgent

@atheris.instrument_func
async def main(input_message):
    rag = RAGAgent()
    await rag.CreateGraph()
    config = RunnableConfig(run_name="RAG Deep Agent", configurable={"thread_id": uuid7str(), "user_id": uuid7str()})
    await rag.ChatAgent(config, input_message, ["values"], False)

@atheris.instrument_func
def FuzzEntryPoint(data):
    # Initialize the provider with raw bytes from the fuzzer
    fdp = atheris.FuzzedDataProvider(data)    
    # Consume structured data
    number = fdp.ConsumeIntInRange(100, 100000)
    input_message = fdp.ConsumeUnicodeNoSurrogates(128)
    asyncio.run(main(input_message))

if __name__ == "__main__":
    atheris.Setup(sys.argv, FuzzEntryPoint)
    atheris.instrument_all()    
    atheris.Fuzz()    
