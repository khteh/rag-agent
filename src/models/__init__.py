from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from quart_bcrypt import Bcrypt
from src.config import config
from .schema import UserInput, AgentResponse, ChatMessage, StreamInput
db = SQLAlchemy()
bcrypt = Bcrypt()
engine = create_engine(config.SQLALCHEMY_DATABASE_URI)
__all__ = ["UserInput", "AgentResponse", "ChatMessage", "StreamInput", "db", "bcrypt", "engine"]
