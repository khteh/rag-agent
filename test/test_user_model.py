import pytest, pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from src.models.base import Base
import src.models.UserModel as user_module
from src.models.UserModel import UserModel, UserSchema

"""
Exercises the async SQLAlchemy user layer against an in-memory SQLite database (aiosqlite), so no PostgreSQL is needed.
"""
@pytest_asyncio.fixture(scope="function")
async def sqlite_sessions(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    monkeypatch.setattr(user_module, "SessionLocal", async_sessionmaker(engine, expire_on_commit=False))
    yield
    await engine.dispose()

def new_user(email="jane@example.com"):
    # Same path as the controllers: marshmallow builds a transient instance, no database session required.
    return UserSchema().load({"firstname": "Jane", "lastname": "Doe", "email": email, "password": "s3cret!"})

async def test_add_and_get(sqlite_sessions):
    id = await UserModel.add(new_user())
    assert id
    user = await UserModel.get_user(id)
    assert user.email == "jane@example.com"
    assert user.check_hash("s3cret!")
    assert user.password != "s3cret!"
    assert user.created_at is not None # server_default is readable after the session closed

async def test_existing_user_and_lookup(sqlite_sessions):
    assert not await UserModel.isExistingUser("jane@example.com")
    await UserModel.add(new_user())
    assert await UserModel.isExistingUser("jane@example.com")
    assert not await UserModel.isExistingUser("other@example.com")
    assert (await UserModel.get_user_by_email("jane@example.com")).firstname == "Jane"
    assert await UserModel.get_user_by_email("other@example.com") is None
    assert len(await UserModel.get_users()) == 1

async def test_update_persists(sqlite_sessions):
    id = await UserModel.add(new_user())
    user = await UserModel.get_user(id)
    changes = UserSchema().load({"firstname": "Janet"}, partial=True)
    await user.update(changes)
    assert user.firstname == "Janet" # instance stays readable after the session closed
    assert (await UserModel.get_user(id)).firstname == "Janet" # and the change really reached the database

async def test_delete(sqlite_sessions):
    id = await UserModel.add(new_user())
    user = await UserModel.get_user(id)
    await user.delete()
    assert await UserModel.get_user(id) is None
    assert not await UserModel.isExistingUser("jane@example.com")
