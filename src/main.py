import quart_flask_patch
import logging, os, re, json, asyncio, psycopg
from dotenv import load_dotenv
from datetime import datetime
from hypercorn.config import Config
from hypercorn.asyncio import serve
from src.app import create_app
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
load_dotenv()
config = Config()
#from .common.Authentication import oidc
config.from_toml("/etc/pythonrestapi.toml")
app = create_app()
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')	
#oidc.init_app(app)
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}
config = RunnableConfig(run_name="RAG ReAct Agent", thread_id=datetime.now())
from src.rag_agent.RAGAgent import make_graph
agent = asyncio.get_event_loop().run_until_complete(make_graph(config))

# https://quart.palletsprojects.com/en/latest/how_to_guides/startup_shutdown.html
@app.while_serving
async def lifespan():
    print(f"\n=== {lifespan.__name__} ===")
    async with AsyncConnectionPool(
        conninfo=app.config["SQLALCHEMY_DATABASE_URI"],
        max_size=app.config["DB_MAX_CONNECTIONS"],
        kwargs=connection_kwargs,
    ) as pool:
        # Create the AsyncPostgresSaver
        checkpointer = AsyncPostgresSaver(pool)
        
        # Set up the checkpointer (uncomment this line the first time you run the app)
        #await checkpointer.setup()
        # Check if the checkpoints table exists
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                try:
                    await cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE  table_schema = 'public'
                            AND    table_name   = 'checkpoints'
                        );
                    """)
                    table_exists = (await cur.fetchone())[0]
                    
                    if not table_exists:
                        print("Checkpoints table does not exist. Running setup...")
                        await checkpointer.setup()
                    else:
                        print("Checkpoints table already exists. Skipping setup.")
                except psycopg.Error as e:
                    print(f"Error checking for checkpoints table: {e}")
                    # Optionally, you might want to raise this error
                    # raise
        
        # Assign the checkpointer to the assistant
        agent.checkpointer = checkpointer
        app.state.agent = agent
        yield

#app.run(HOST, PORT)
print(f"Running asyncio...")
asyncio.run(serve(app, config), debug=True)