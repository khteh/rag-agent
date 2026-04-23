import atheris, asyncio, sys
from src.config import config
from data.sample_emails import EMAILS
from src.models import ChatMessage
from src.EmailRAG.EmailRAG import EmailRAG
from psycopg_pool import AsyncConnectionPool
db_pool = AsyncConnectionPool(
                conninfo = config.POSTGRESQL_DATABASE_URI,
                max_size = config.DB_MAX_CONNECTIONS,
                kwargs = config.connection_kwargs,
                open = False # RuntimeError: AsyncConnectionPool open with no running loop if set to True
            )
# rag = RAGAgent(db_pool)  ChatAgent exception! Event loop is closed, repr: RuntimeError('Event loop is closed')
# ChatAgent exception! <asyncio.locks.Lock object at 0x7ce41dbdf460 [locked]> is bound to a different event loop, repr: RuntimeError('<asyncio.locks.Lock object at 0x7ce41dbdf460 [locked]> is bound to a different event loop')
async def main(criteria):
    # httpx library is a dependency of LangGraph and is used under the hood to communicate with the AI models.
    """
    graph = checkpoint_graph.get_graph().draw_mermaid_png()
    # Save the PNG data to a file
    with open("/tmp/checkpoint_graph.png", "wb") as f:
        f.write(graph)
    img = Image.open("/tmp/checkpoint_graph.png")
    img.show()
    """
    rag = EmailRAG(db_pool)
    await rag.CreateGraph()
    email_state = {
        "email": EMAILS[3],
        "escalation_dollar_criteria": 100_000,
        "escalation_emails": ["brog@abc.com", "bigceo@company.com"],
    }
    result = await rag.Chat(criteria, email_state)
    assert result
    ai_message = ChatMessage.from_langchain(result)
    assert not ai_message.tool_calls
    assert ai_message.content

def FuzzEntryPoint(data):
    # Initialize the provider with raw bytes from the fuzzer
    fdp = atheris.FuzzedDataProvider(data)    
    # Consume structured data
    number = fdp.ConsumeIntInRange(100, 100000)
    criteria = fdp.ConsumeUnicodeNoSurrogates(128)
    print(f"criteria: {repr(criteria)}", flush=True)
    asyncio.run(main(criteria))

if __name__ == "__main__":
    atheris.instrument_all()
    atheris.Setup(sys.argv, FuzzEntryPoint)
    atheris.Fuzz()    
