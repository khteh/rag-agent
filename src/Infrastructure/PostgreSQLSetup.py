import logging
from psycopg import Error
from langchain_ollama import OllamaEmbeddings
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from src.config import config as appconfig
async def PostgreSQLCheckpointerSetup(pool: AsyncConnectionPool) -> AsyncPostgresSaver:
    # Set up the checkpointer (uncomment this line the first time you run the app)
    # Check if the checkpoints table exists
    checkpointer = AsyncPostgresSaver(pool)
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
                    logging.info("Checkpoints table does not exist. Running setup...")
                    await checkpointer.setup()
                else:
                    logging.info("Checkpoints table already exists. Skipping setup.")
            except Error as e:
                logging.exception(f"Error checking for checkpoints table: {e}")
                # Optionally, you might want to raise this error
                raise e
    return checkpointer

async def PostgreSQLStoreSetup(pool: AsyncConnectionPool) -> AsyncPostgresStore:
    # Set up the checkpointer (uncomment this line the first time you run the app)
    # Check if the checkpoints table exists
    logging.debug(f"EMBEDDING_DIMENSIONS: {appconfig.EMBEDDING_DIMENSIONS}")
    store = AsyncPostgresStore(pool, index={
                "embed": OllamaEmbeddings(model=appconfig.EMBEDDING_MODEL, base_url=appconfig.BASE_URI, num_ctx=8192, num_gpu=1, temperature=0),
                "dims": appconfig.EMBEDDING_DIMENSIONS, # Note: Every time when this value changes, remove the store<foo> tables in the DB so that store.setup() runs to recreate them with the right dimensions.
            }
    )
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            try:
                await cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE  table_schema = 'public'
                        AND    table_name   = 'store'
                    );
                """)
                table_exists = (await cur.fetchone())[0]
                if not table_exists:
                    logging.info("store table does not exist. Running setup...")
                    await store.setup()  # Run migrations. Done once
                else:
                    logging.info("store table already exists. Skipping setup.")
            except Error as e:
                logging.exception(f"Error checking for store table: {e}")
                # Optionally, you might want to raise this error
                raise e
    return store