from datetime import datetime
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema, auto_field
from sqlalchemy import Integer, String, DateTime, select
from marshmallow import fields, Schema
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column, Session
from src.common.Bcrypt import bcrypt
from .base import Base
from . import engine
class UserModel(Base):
    """
    User Model
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
    def save(self):
        with Session(engine) as session:
            session.commit()
    def update(self, data):
        for attribute, value in vars(data).items():
            if attribute == "password":
                value = self.__generate_hash(value)
            setattr(self, attribute, value)
        self.modified_at = func.now()
        with Session(engine) as session:
            session.commit()
    def delete(self):
        with Session(engine) as session:
            session.delete(self)
            session.commit()
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
    def add(user) -> int:
        with Session(engine) as session:
            user.password = user.__generate_hash(user.password)
            session.add(user)
            # Flush the session
            # This operation sends the INSERT statement to the database.
            # The database generates the primary key and returns it to SQLAlchemy,
            # which updates the 'id' attribute of the 'new_user' object *in memory*.
            session.flush()
            session.commit()
            return user.id
    @staticmethod
    def get_user(id):
        # Usage and parameters are the same as that of Session.execute(); the return result is a ScalarResult filtering object which will return single elements rather than Row objects.
        with Session(engine) as session:
            return session.get(UserModel, id)
    @staticmethod
    def get_user_by_email(email):
        stmt = select(UserModel).filter_by(email=email).execution_options(populate_existing=True)
        with Session(engine) as session:
            return session.scalars(stmt).first()
    @staticmethod
    def isExistingUser(email):
        with Session(engine) as session:
            return session.query(UserModel).filter_by(email=email).count() > 0
    @staticmethod
    def get_users():
        stmt = select(UserModel).execution_options(populate_existing=True)
        # Usage and parameters are the same as that of Session.execute(); the return result is a ScalarResult filtering object which will return single elements rather than Row objects.
        with Session(engine) as session:
            return session.scalars(stmt).all()
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
    id = fields.Int(dump_only=True)
    firstname = fields.Str(required=True)
    lastname = fields.Str(required=True)
    email = fields.Email(required=True)
    phone = fields.Str(required=False)
    password = fields.Str(required=True)
    lastlogin = fields.DateTime(required=False)
    created_at = fields.DateTime(dump_only=True)
    modified_at = fields.DateTime(dump_only=True)