from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import AsyncConnectionPool, ConnectionPool
from src.config import config
_connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}
async def GetConnectionPool():
    async with AsyncConnectionPool(
        conninfo = config.POSTGRESQL_DATABASE_URI,
        max_size = config.DB_MAX_CONNECTIONS,
        kwargs = _connection_kwargs,
    ) as pool:
        return pool
    
async def GetAsyncCheckpointer():
    pool = AsyncConnectionPool(
        conninfo = config.POSTGRESQL_DATABASE_URI,
        max_size = config.DB_MAX_CONNECTIONS,
        kwargs = _connection_kwargs,
    )
    # Create the AsyncPostgresSaver
    return AsyncPostgresSaver(pool)
    
def GetCheckpointer():
    return PostgresSaver(ConnectionPool(
        conninfo = config.POSTGRESQL_DATABASE_URI,
        max_size = config.DB_MAX_CONNECTIONS,
        kwargs = _connection_kwargs,
    ))
