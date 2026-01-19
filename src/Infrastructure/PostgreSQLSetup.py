import logging
from psycopg import Error
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
async def PostgreSQLCheckpointerSetup(pool: AsyncConnectionPool, checkpointer: AsyncPostgresSaver) -> None:
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

async def PostgreSQLStoreSetup(pool: AsyncConnectionPool, store: AsyncPostgresStore) -> None:
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