"""
This script runs the application using a development server.
It contains the definition of routes and views for the application.
"""
import quart_flask_patch
import json, logging, os, vertexai, psycopg
from quart import Quart, request
from flask_healthz import HealthError
from flask_bcrypt import Bcrypt
from flask_healthz import Healthz
from quart_wtf.csrf import CSRFProtect
from quart_cors import cors
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.graph import CompiledGraph
from psycopg_pool import AsyncConnectionPool
from src.controllers.HomeController import home_api as home_blueprint
bcrypt = Bcrypt()
# Make the WSGI interface available at the top level so wfastcgi can get it.
app = None
def create_app() -> Quart:
    """
    Create App
    """
    # App initialization
    #print(f"project={os.environ.get('GOOGLE_CLOUD_PROJECT')}, location={os.environ.get('GOOGLE_CLOUD_LOCATION')}")
    vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    app = Quart(__name__, static_url_path='')
    app.config.from_file("/etc/rag-agent_config.json", json.load)
    if "SQLALCHEMY_DATABASE_URI" not in app.config:
        app.config["SQLALCHEMY_DATABASE_URI"] = f"postgresql://{os.getenv('POSTGRESQL_USER')}:{os.getenv('POSTGRESQL_PASSWORD')}@svc-postgresql/rag-agent"
    app = cors(app, allow_credentials=True, allow_origin="https://localhost:4433")
    app.register_blueprint(home_blueprint, url_prefix="/")
    Healthz(app, no_log=True)
    csrf = CSRFProtect(app)
    bcrypt.init_app(app)
    return app

def liveness():
    pass 

async def readiness():
    try:
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
        }
        async with AsyncConnectionPool(
            conninfo=app.config["SQLALCHEMY_DATABASE_URI"],
            #max_size=DB_MAX_CONNECTIONS,
            kwargs=connection_kwargs,
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
                    except psycopg.Error as e:
                        raise HealthError(f"Error checking for checkpoints table: {e}")
                        # Optionally, you might want to raise this error
                        # raise
    except Exception:
        raise HealthError(f"Failed to connect to the database!")