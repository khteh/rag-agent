from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from quart_bcrypt import Bcrypt
from src.config import config
from .schema import UserInput, AgentResponse, ChatMessage, StreamInput
bcrypt = Bcrypt()
# postgresql+psycopg:// is served by psycopg 3, which supports both sync and async. Alembic keeps using its own sync engine (see migrations/env.py).
# used only by UserModel, through SQLAlchemy async on the main event loop.
# pool_size=2, max_overflow=3	max connections: 5
# At startup: create_async_engine(...) opens nothing. The pool is created empty.
# First use: the first time code runs async with SessionLocal() and a query (for example, the first login), the pool opens a connection.
# After use: when the session closes, the connection is returned to the pool and stays open for reuse, up to pool_size. It is only closed at engine.dispose(), by pool_recycle if you set it, or when pool_pre_ping finds it dead.
engine = create_async_engine(config.SQLALCHEMY_DATABASE_URI, pool_pre_ping=True, pool_size=1, max_overflow=2)
# expire_on_commit=False keeps loaded attributes readable after the session is closed. Lazy loading is not possible in async sessions.
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
__all__ = ["UserInput", "AgentResponse", "ChatMessage", "StreamInput", "bcrypt", "engine", "SessionLocal"]
