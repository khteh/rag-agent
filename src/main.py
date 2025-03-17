import quart_flask_patch
import logging, os, re, json, asyncio, psycopg, json, logging, vertexai
from dotenv import load_dotenv
from datetime import datetime
from hypercorn.config import Config
from hypercorn.asyncio import serve
from psycopg_pool import AsyncConnectionPool
from datetime import datetime
from quart import Quart, request
from flask_healthz import Healthz, HealthError
from flask_bcrypt import Bcrypt
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
load_dotenv()
#print(f"GEMINI_API_KEY: {os.environ.get("GEMINI_API_KEY")}")
config = Config()
config.from_toml("/etc/hypercorn.toml")
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')	
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}
async def create_app() -> Quart:
    """
    Create App
    """
    # App initialization
    vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    app = Quart(__name__, static_url_path='')
    app.config.from_file("/etc/ragagent_config.json", json.load)
    if "SQLALCHEMY_DATABASE_URI" not in app.config:
        app.config["SQLALCHEMY_DATABASE_URI"] = f"postgresql://{os.getenv('POSTGRESQL_USER')}:{os.getenv('POSTGRESQL_PASSWORD')}@{app.config['DB_HOST']}/rag-agent"
    app = cors(app, allow_credentials=True, allow_origin="https://localhost:4433")
    app.register_blueprint(home_blueprint, url_prefix="/")
    Healthz(app, no_log=True)
    # https://quart-wtf.readthedocs.io/en/stable/how_to_guides/configuration.html
    csrf = CSRFProtect(app)
    bcrypt.init_app(app)
    config = RunnableConfig(run_name="RAG ReAct Agent", thread_id=datetime.now())
    from src.rag_agent.RAGAgent import make_graph, agent
    agent = await make_graph(config)
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
        if agent:
            agent.checkpointer = checkpointer
            print("Agent is assigned a checkpointer")
        else:
            print(f"Agent not ready")
    return app

# https://quart.palletsprojects.com/en/latest/how_to_guides/startup_shutdown.html
def liveness():
    print("liveness")
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
        print("Ready!")
    except Exception:
        raise HealthError(f"Failed to connect to the database!")

app = asyncio.get_event_loop().run_until_complete(create_app())
print(f"Running asyncio...")
#asyncio.run(serve(app, config), debug=True)