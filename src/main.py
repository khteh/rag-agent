import logging, os, re, json, asyncio, psycopg, json
from uuid_extensions import uuid7, uuid7str
from datetime import date, datetime, timedelta, timezone
from hypercorn.config import Config
from psycopg_pool import AsyncConnectionPool, ConnectionPool
from quart import Quart, Response, json, Blueprint, session, render_template, session, redirect, url_for, flash
from src.common.Bcrypt import bcrypt
from quart_wtf.csrf import CSRFProtect, CSRFError
from quart_cors import cors
from langchain_core.runnables import RunnableConfig
from src.config import config as appconfig
from src.Infrastructure.PostgreSQLSetup import PostgreSQLCheckpointerSetup
import warnings
"""
https://docs.python.org/3/library/warnings.html#warnings.filterwarnings
https://docs.python.org/3/library/warnings.html#warning-filter
"""
warnings.filterwarnings('always')
# Make the WSGI interface available at the top level so wfastcgi can get it.
#print(f"GEMINI_API_KEY: {os.environ.get("GEMINI_API_KEY")}")
config = Config()
config.from_toml("/etc/hypercorn.toml")

# httpx library is a dependency of LangGraph and is used under the hood to communicate with the AI models.
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
    logging.debug(f"\n=== {create_app.__name__} ===")
    #vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    app = Quart(__name__, template_folder='view/templates', static_url_path='', static_folder='view/static')
    app.config.from_file("/etc/ragagent_config.json", json.load)
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = timedelta(days=90)
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config["WTF_CSRF_TIME_LIMIT"] = None # Max age in seconds for CSRF tokens. If set to None, the CSRF token is valid for the life of the session.
    app = cors(app, allow_credentials=True, allow_origin="https://localhost:4433")

    @app.errorhandler(CSRFError)
    async def handle_csrf_error(e):
        logging.exception(f"handle_csrf_error {e.description}")
        await flash("Session expired!", "danger")
        if "url" in session and session["url"]:
            return redirect(session["url"]), 440
        else:
            return redirect(url_for("home.index")), 440

    @app.before_serving
    async def before_serving() -> None:
        logging.debug(f"\n=== {before_serving.__name__} ===")
        app.db_pool = AsyncConnectionPool(
            conninfo = appconfig.POSTGRESQL_DATABASE_URI,
            max_size = appconfig.DB_MAX_CONNECTIONS,
            kwargs = appconfig.connection_kwargs,
        )
        await app.db_pool.open()
        # Create the AsyncPostgresSaver
        checkpointer = await PostgreSQLCheckpointerSetup(app.db_pool)
        # Assign the checkpointer to the assistant
        app.agent.checkpointer = checkpointer
        app.graph_rag.checkpointer = checkpointer
        app.healthcare_agent.checkpointer = checkpointer
        logging.debug(f"\n=== {before_serving.__name__} done! ===")

    @app.after_serving
    async def after_serving():
        logging.debug(f"\n=== {after_serving.__name__} ===")
        await app.db_pool.close()

    app.after_request(_add_secure_headers)
    from src.controllers.HomeController import home_api as home_blueprint
    from src.controllers.HealthController import health_api as health_blueprint
    from src.controllers.HealthcareController import healthcare_api as healthcare_blueprint
    app.register_blueprint(home_blueprint, url_prefix="/")
    app.register_blueprint(health_blueprint, url_prefix="/health")
    app.register_blueprint(healthcare_blueprint, url_prefix="/healthcare")
    # https://quart-wtf.readthedocs.io/en/stable/how_to_guides/configuration.html
    CSRFProtect(app)
    bcrypt.init_app(app)
    from src.rag_agent.RAGAgent import make_graph
    from src.rag_agent.GraphRAG import make_graph as graph_rag_make_graph
    from src.Healthcare.RAGAgent import make_graph as healthcare_make_graph
    config = RunnableConfig(run_name="RAG Deep Agent", thread_id=uuid7str())
    grapg_rag_config = RunnableConfig(run_name="RAG Deep Agent", thread_id=uuid7str())
    healthcare_config = RunnableConfig(run_name="Healthcare Sub-Agent", thread_id=uuid7str())
    app.agent = await make_graph(config)
    app.graph_rag = await graph_rag_make_graph(grapg_rag_config)
    app.healthcare_agent = await healthcare_make_graph(healthcare_config)
    #if app.debug:
    # https://github.com/pgjones/hypercorn/issues/294
    #    return HTTPToHTTPSRedirectMiddleware(app, "khteh.com")  # type: ignore - Defined in hypercorn.toml server_names
    #else:
    logging.debug(f"\n=== {create_app.__name__} done! ===")
    return app

app = asyncio.run(create_app())

logging.info(f"Running app...")
#asyncio.run(serve(app, config), debug=True)