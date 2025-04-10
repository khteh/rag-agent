import logging, os, re, json, asyncio, psycopg, json, vertexai
from uuid_extensions import uuid7, uuid7str
from urllib import parse
from datetime import date, datetime, timedelta, timezone
from hypercorn.config import Config
from hypercorn.asyncio import serve
from psycopg import Error
from psycopg_pool import AsyncConnectionPool, ConnectionPool
from quart import Quart, Response
from src.common.Bcrypt import bcrypt
from quart_wtf.csrf import CSRFProtect
from quart_cors import cors
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.graph import CompiledGraph
from src.config import config as appconfig
# Make the WSGI interface available at the top level so wfastcgi can get it.
#print(f"GEMINI_API_KEY: {os.environ.get("GEMINI_API_KEY")}")
config = Config()
config.from_toml("/etc/hypercorn.toml")
# httpx library is a dependency of LangGraph and is used under the hood to communicate with the AI models.
"""
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}
"""
def _add_secure_headers(response: Response) -> Response:
    response.headers["Strict-Transport-Security"] = (
        "max-age=63072000; includeSubDomains; preload"
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response

async def create_app() -> Quart:
    """
    Create App
    """
    # App initialization
    #vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    app = Quart(__name__, template_folder='view/templates', static_url_path='', static_folder='view/static')
    app.config.from_file("/etc/ragagent_config.json", json.load)
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = timedelta(days=90)
    app.config["SQLALCHEMY_DATABASE_URI"] = f"postgresql+psycopg://{os.environ.get('DB_USERNAME')}:{parse.quote(os.environ.get('DB_PASSWORD'))}@{app.config['DB_HOST']}/LangchainCheckpoint"
    app.config["POSTGRESQL_DATABASE_URI"] = f"postgresql://{os.environ.get('DB_USERNAME')}:{parse.quote(os.environ.get('DB_PASSWORD'))}@{app.config['DB_HOST']}/LangchainCheckpoint"
    app.after_request(_add_secure_headers)
    app = cors(app, allow_credentials=True, allow_origin="https://localhost:4433")
    from src.controllers.HomeController import home_api as home_blueprint
    from src.controllers.HealthController import health_api as health_blueprint
    app.register_blueprint(home_blueprint, url_prefix="/")
    app.register_blueprint(health_blueprint, url_prefix="/health")
    # https://quart-wtf.readthedocs.io/en/stable/how_to_guides/configuration.html
    csrf = CSRFProtect(app)
    bcrypt.init_app(app)
    config = RunnableConfig(run_name="RAG ReAct Agent", thread_id=uuid7str())
    healthcare_config = RunnableConfig(run_name="Healthcare ReAct Agent", thread_id=uuid7str())
    from src.rag_agent.RAGAgent import make_graph#, agent
    from src.Healthcare.RAGAgent import make_graph as healthcare_make_graph#, agent
    app.agent = await make_graph(config)
    app.healthcare_agent = await healthcare_make_graph(healthcare_config)
    if app.debug:
        return HTTPToHTTPSRedirectMiddleware(app, "khteh.com")  # type: ignore - Defined in hypercorn.toml server_names
    else:
        app.config["TEMPLATES_AUTO_RELOAD"] = True
    @app.before_serving
    async def startup() -> None:
        logging.debug(f"\n=== {startup.__name__} ===")
        async with AsyncConnectionPool(
            conninfo = app.config["POSTGRESQL_DATABASE_URI"],
            max_size = appconfig.DB_MAX_CONNECTIONS,
            kwargs = appconfig.connection_kwargs,
        ) as pool:
            # Create the AsyncPostgresSaver
            checkpointer = AsyncPostgresSaver(pool)
            # Set up the checkpointer (uncomment this line the first time you run the app)
            logging.INFO("checkpointer setup...")
            await checkpointer.setup()
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
                            logging.info("Checkpoints table does not exist. Running setup...")
                            await checkpointer.setup()
                        else:
                            logging.info("Checkpoints table already exists. Skipping setup.")
                    except psycopg.Error as e:
                        logging.exception(f"Error checking for checkpoints table: {e}")
                        # Optionally, you might want to raise this error
                        # raise
            # Assign the checkpointer to the assistant
            app.agent.checkpointer = checkpointer
            app.healthcare_agent.checkpointer = checkpointer
        logging.debug(f"\n=== {startup.__name__} done! ===")
    return app

app = asyncio.get_event_loop().run_until_complete(create_app())
logging.info(f"Running app...")
#asyncio.run(serve(app, config), debug=True)