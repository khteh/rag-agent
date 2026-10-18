from datetime import datetime
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema, auto_field
from sqlalchemy import Integer, String, DateTime, select
from marshmallow import fields, Schema
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column
from src.common.Bcrypt import bcrypt
from .base import Base
from . import SessionLocal
class UserModel(Base):
    """
    User Model
    https://marshmallow-sqlalchemy.readthedocs.io/en/latest/
    """
    __tablename__ = 'users'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    firstname: Mapped[str] = mapped_column(String(128), nullable=False)
    lastname: Mapped[str] = mapped_column(String(128), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    phone: Mapped[str] = mapped_column(String(15), unique=True, nullable=True, index=True)
    password: Mapped[str] = mapped_column(String(128), nullable=True)
    lastlogin: Mapped[DateTime] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    modified_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    @classmethod
    def fromdata(cls, data):
        user = UserModel()
        user.firstname = data.firstname
        user.lastname = data.lastname
        user.email = data.email
        user.phone = data.phone
        user.password = user.__generate_hash(data.password)
        user.created_at = func.now()
        user.modified_at = func.now()
        return user
    async def save(self):
        async with SessionLocal() as session:
            session.add(self)
            await session.commit()
            await session.refresh(self)
    async def update(self, data):
        async with SessionLocal() as session:
            session.add(self) # Re-attach this detached instance to the new session
            for attribute, value in vars(data).items():
                if attribute.startswith("_"): # Skip SQLAlchemy internals such as _sa_instance_state
                    continue
                if attribute == "password":
                    value = self.__generate_hash(value)
                setattr(self, attribute, value)
            self.modified_at = func.now()
            await session.commit()
            # modified_at is a server-side expression; reload so the detached instance stays readable.
            await session.refresh(self)
    async def delete(self):
        async with SessionLocal() as session:
            await session.delete(self)
            await session.commit()
    def __generate_hash(self, password):
        return bcrypt.generate_password_hash(password, rounds=10).decode("utf-8")
    def check_hash(self, password):
        return bcrypt.check_password_hash(self.password, password)
    @property
    def serialized(self):
        """Return object data in serializable format"""
        return {
            'id': self.id,
            'firstname': self.firstname,
            'lastname': self.lastname,
            'email': self.email,
            'phone': self.phone,
            'lastlogin': self.lastlogin,
            'created_at': self.created_at,
            'modified_at': self.modified_at
        }
    @staticmethod
    async def add(user) -> int:
        async with SessionLocal() as session:
            user.password = user.__generate_hash(user.password)
            session.add(user)
            # Flush the session
            # This operation sends the INSERT statement to the database.
            # The database generates the primary key and returns it to SQLAlchemy,
            # which updates the 'id' attribute of the 'new_user' object *in memory*.
            await session.flush()
            await session.commit()
            return user.id
    @staticmethod
    async def get_user(id):
        async with SessionLocal() as session:
            return await session.get(UserModel, id)
    @staticmethod
    async def get_user_by_email(email):
        stmt = select(UserModel).filter_by(email=email).execution_options(populate_existing=True)
        async with SessionLocal() as session:
            # Usage and parameters are the same as that of AsyncSession.execute(); the return result is a ScalarResult filtering object which will return single elements rather than Row objects.
            return (await session.scalars(stmt)).first()
    @staticmethod
    async def isExistingUser(email):
        stmt = select(func.count()).select_from(UserModel).where(UserModel.email == email)
        async with SessionLocal() as session:
            return (await session.scalar(stmt)) > 0
    @staticmethod
    async def get_users():
        stmt = select(UserModel).execution_options(populate_existing=True)
        async with SessionLocal() as session:
            return (await session.scalars(stmt)).all()
    def __repl__(self): # return a printable representation of UserModel object, in this case we're only returning the id
        return "<id {}>".format(self.id)
class UserSchema(SQLAlchemyAutoSchema):
    """
    User Schema
    """
    class Meta:
        model = UserModel
        # Optional: include relationships, foreign keys, etc.
        include_relationships = True
        # Optional: deserialize to model instances
        load_instance = True
        # Build new instances without a database session (AsyncSession cannot be used by marshmallow-sqlalchemy's sync lookups)
        transient = True
    id = fields.Int(dump_only=True)
    firstname = fields.Str(required=True)
    lastname = fields.Str(required=True)
    email = fields.Email(required=True)
    phone = fields.Str(required=False)
    password = fields.Str(required=True)
    lastlogin = fields.DateTime(required=False)
    created_at = fields.DateTime(dump_only=True)
    modified_at = fields.DateTime(dump_only=True)