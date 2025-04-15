import logging
from uuid_extensions import uuid7, uuid7str
from typing import AsyncGenerator, Dict, Any, Tuple
from langchain_core.runnables import RunnableConfig
from src.models.schema import ChatMessage, UserInput, StreamInput
def parse_input(user_input: UserInput) -> Tuple[Dict[str, Any], str]:
    run_id = uuid7()
    logging.debug(f"user_input: {user_input}")
    thread_id = "thread_id" in user_input and user_input.thread_id or uuid7str()
    user_id = "user_id" in user_input and user_input.user_id or uuid7str()
    input_message = ChatMessage(type="human", content=user_input["message"])
    config = RunnableConfig(run_name="RAG ReAct Agent", thread_id=thread_id, user_id=user_id)
    kwargs = dict(
        input={"messages": [input_message.to_langchain()]},
        config = config,
    )
    return kwargs, run_id

