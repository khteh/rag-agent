import chromadb, re, jsonpickle, logging
from chromadb.config import Settings
from psycopg_pool import AsyncConnectionPool, ConnectionPool
from quart import (
    Blueprint,
    Response,
    ResponseReturnValue,
    current_app,
    make_response,
    render_template,
    session
)
from src.config import config
health_api = Blueprint("health", __name__)

async def _healthcheck() -> ResponseReturnValue:
    try:
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
        }
        async with AsyncConnectionPool(
            conninfo = current_app.config["POSTGRESQL_DATABASE_URI"],
            max_size = current_app.config["DB_MAX_CONNECTIONS"],
            kwargs = connection_kwargs,
        ) as pool:
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
                    except Exception as e:
                        logging.exception(f"Error checking for checkpoints table: Exception: {e}")
                        raise e
        client = chromadb.HttpClient(host=config.CHROMA_URI,
                                settings=Settings(
                                    chroma_client_auth_provider = "chromadb.auth.token_authn.TokenAuthenticationServerProvider",
                                    chroma_client_auth_credentials = config.CHROMA_TOKEN,
                                    chroma_auth_token_transport_header = "X-Chroma-Token"
                                ))
        client.heartbeat()  # this should work with or without authentication - it is a public endpoint
        logging.debug("Healthcheck OK!")
        return "OK", 200
    except Exception:
        logging.exception(f"{readiness.__name__} Exception: {e}")
    return "Exceptions!", 500

@health_api.route("/ready")
async def readiness() -> ResponseReturnValue:
    return await _healthcheck()

@health_api.route("/live")
async def liveness():
    return await _healthcheck()  

