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
from src.Infrastructure.Checkpointer import CheckpointerSetup
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
    app = cors(app, allow_credentials=True, allow_origin="https://localhost:4433")
    @app.before_serving
    async def before_serving() -> None:
        logging.debug(f"\n=== {before_serving.__name__} ===")
        app.db_pool = AsyncConnectionPool(
            conninfo = app.config["POSTGRESQL_DATABASE_URI"],
            max_size = appconfig.DB_MAX_CONNECTIONS,
            kwargs = appconfig.connection_kwargs,
        )
        # Create the AsyncPostgresSaver
        checkpointer = await CheckpointerSetup(app.db_pool)
        # Assign the checkpointer to the assistant
        app.agent.checkpointer = checkpointer
        app.healthcare_agent.checkpointer = checkpointer
        logging.debug(f"\n=== {before_serving.__name__} done! ===")

    @app.after_serving
    async def after_serving():
        logging.debug(f"\n=== {after_serving.__name__} ===")
        await app.db_pool.close()

    app.after_request(_add_secure_headers)
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
    return app

app = asyncio.get_event_loop().run_until_complete(create_app())


logging.info(f"Running app...")
#asyncio.run(serve(app, config), debug=True)