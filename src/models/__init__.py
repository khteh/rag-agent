from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from quart_bcrypt import Bcrypt
from src.config import config
from .schema import UserInput, AgentResponse, ChatMessage, StreamInput
bcrypt = Bcrypt()
# postgresql+psycopg:// is served by psycopg 3, which supports both sync and async. Alembic keeps using its own sync engine (see migrations/env.py).
engine = create_async_engine(config.SQLALCHEMY_DATABASE_URI, pool_pre_ping=True)
# expire_on_commit=False keeps loaded attributes readable after the session is closed. Lazy loading is not possible in async sessions.
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
__all__ = ["UserInput", "AgentResponse", "ChatMessage", "StreamInput", "bcrypt", "engine", "SessionLocal"]
