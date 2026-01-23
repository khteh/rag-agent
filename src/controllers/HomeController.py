import re, asyncio, json, jsonpickle, logging, jsonpickle, pickle
from uuid_extensions import uuid7, uuid7str
from typing import AsyncGenerator, Dict, Any, Tuple
from quart import (
    Blueprint,
    flash,
    formparser,
    request,
    Response,
    ResponseReturnValue,
    current_app,
    make_response,
    render_template,
    session
)
from datetime import datetime, timezone
from werkzeug.exceptions import HTTPException
from quart.helpers import stream_with_context
from src.common.Authentication import Authentication
from src.models.schema import ChatMessage, UserInput, StreamInput
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, ToolCall
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.runnables import RunnableConfig
from urllib.parse import urlparse, parse_qs
from typing_extensions import List, TypedDict
from src.common.ResponseHelper import Respond
from src.common.Response import custom_response
from src.utils.JsonString import is_json
home_api = Blueprint("home", __name__)
@home_api.context_processor
def inject_now():
    return {'now': datetime.now(timezone.utc)}

@home_api.route("/")
@home_api.route("/index")
async def index() -> ResponseReturnValue:
    greeting = None
    now = datetime.now()
    # https://www.programiz.com/python-programming/datetime/strftime
    formatted_now = now.strftime("%A, %d %B, %Y at %X")
    #if request.method == "POST":
    user = None
    if "user" not in session or not session["user"]:
        greeting = "Friend! It's " + formatted_now
        #print(f"homeController hello greeting: {greeting}")
        return await Respond("index.html", title="Welcome to LLM-RAG 💬", greeting=greeting)
    user = jsonpickle.decode(session['user'])
    if not user or not hasattr(user, 'token'):
        greeting = "Friend! It's " + formatted_now
        #print(f"homeController hello greeting: {greeting}")
        return await Respond("index.html", title="Welcome to LLM-RAG 💬", greeting=greeting)
    data = Authentication.decode_token(user.token)
    if data["error"]:
        return await Respond("login.html", title="Welcome to LLM-RAG 💬", error=data["error"])
    user_id = data["data"]["user_id"]
    logging.debug(f"User: {user_id}")
    """
    user = UserModel.get_user(user_id)
    if not user:
        return await Respond("login.html", title="Welcome to LLM-RAG 💬", error="Invalid user!")
    try:
        logging.debug(f"Firstname: {user.firstname}, Lastname: {user.lastname}")
        name = user.firstname + ", " + user.lastname
        # Filter the name argument to letters only using regular expressions. URL arguments
        # can contain arbitrary text, so we restrict to safe characters only.
        match_object = re.match("[a-zA-Z ]+", name)
        if match_object:
            clean_name = match_object.group(0)
        else:
            clean_name = "Friend!"
        greeting = f"Hello {clean_name}! It's {formatted_now}"
    except (Exception) as error:
        greeting = "Exception {0}".format(error)
    """
    if not greeting:
        greeting = "Friend! It's " + formatted_now
        #print(f"homeController hello greeting: {greeting}")
    return await Respond("index.html", title="Welcome to LLM-RAG 💬", greeting=greeting)

class TokenQueueStreamingHandler(AsyncCallbackHandler):
    """LangChain callback handler for streaming LLM tokens to an asyncio queue."""
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        if token:
            await self.queue.put(token)

async def ProcessCurlInput() -> UserInput:
    """
    This process curl post body:
    c3 -v https://10.152.183.176/invoke -XPOST -d '{"message": "What is task decomposition?"}'
    """
    data = await request.get_data()
    params = parse_qs(data.decode('utf-8'))
    user_input: UserInput = None
    if is_json(data):
        user_input = json.loads(data)
    elif isinstance(params, dict):
        str_params: str = json.dumps(params)
        if str_params and len(str_params) and is_json(str_params):
            user_input = json.loads(str_params)
    return user_input

@home_api.post("/invoke")
async def invoke():
    """
    Invoke the agent with user input to retrieve a final response.
    Use thread_id to persist and continue a multi-turn conversation.
    """
    if "user_id" not in session or not session["user_id"]:
        session["user_id"] = uuid7str()
    if "thread_id" not in session or not session["thread_id"]:
        session["thread_id"] = uuid7str()
    logging.info(f"\n=== /invoke session: {session['thread_id']}, user_id: {session['user_id']} ===")
    user_input: UserInput = await ProcessCurlInput()
    # If it is not curl, then the request must have come from the browser with form data. The following processes it.
    if not user_input or "message" not in user_input:
        form = await request.form
        #logging.debug(f"form: {form}")
        if "prompt" in form and form["prompt"] and len(form["prompt"]):
            user_input = {"message": form["prompt"]}
    if not user_input or "message" not in user_input or not len(user_input["message"]):
        await flash("Please input your query!", "danger") # https://quart.palletsprojects.com/en/latest/reference/source/quart.helpers.html
        return await Respond("index.html", title="Welcome to LLM-RAG 💬", error="Invalid input!")
    # Expect a single string.
    if isinstance(user_input["message"], (list, tuple)):
        user_input["message"] = user_input["message"][-1]
    config = RunnableConfig(run_name="RAG Deep Agent /invoke", configurable={"thread_id": session["thread_id"], "user_id":  session["user_id"]})
    logging.debug(f"/invoke message: {user_input['message']}")
    result: str = "Oops, there was some error. Please try again!"
    try:
        success, result = await current_app.agent.ChatAgent(config, user_input['message'])
    except Exception as e:
        # https://langchain-ai.github.io/langgraph/troubleshooting/errors/INVALID_CHAT_HISTORY/
        logging.exception(f"/invoke exception! {str(e)}, repr: {repr(e)}")
    return custom_response({"message": result}, 200 if success else 500)

async def message_generator(user_input: StreamInput, config: RunnableConfig) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.
    This is the workhorse method for the /stream endpoint.
    """
    # Use an asyncio queue to process both messages and tokens in
    # chronological order, so we can easily yield them to the client.
    output_queue = asyncio.Queue(maxsize=10)
    if user_input.stream_tokens:
        config["callbacks"] = [TokenQueueStreamingHandler(queue=output_queue)]

    # Pass the agent's stream of messages to the queue in a separate task, so
    # we can yield the messages to the client in the main thread.
    async def run_agent_stream():
        # https://docs.langchain.com/oss/python/langchain/streaming/overview#streaming-from-sub-agents
        async for _, stream_mode, data in current_app.agent.astream(
            {"messages": [{"role": "user", "content": user_input['message']}]},
            stream_mode="updates",
            config = config, # This is needed by Checkpointer
            subgraphs=True,
        ):
            if stream_mode == "updates":
                for source, update in data.items():
                    if source in ("model", "tools"):            
                        await output_queue.put(update["messages"][-1])
        await output_queue.put(None)

    stream_task = asyncio.create_task(run_agent_stream())

    # Process the queue and yield messages over the SSE stream.
    while s := await output_queue.get():
        if isinstance(s, str):
            # str is an LLM token
            yield f"data: {json.dumps({'type': 'token', 'content': s})}\n\n"
            continue

        # Otherwise, s should be a dict of state updates for each node in the graph.
        # s could have updates for multiple nodes, so check each for messages.
        new_messages = []
        for _, state in s.items():
            if "messages" in state:
                new_messages.extend(state["messages"])
        for message in new_messages:
            try:
                chat_message = ChatMessage.from_langchain(message)
                chat_message.thread_id = config.runnable["thread_id"]
                chat_message.user_id = config.runnable["user_id"]
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': f'Error parsing message: {e}'})}\n\n"
                continue
            # LangGraph re-sends the input message, which feels weird, so drop it
            if chat_message.type == "human" and chat_message.content == user_input.message:
                continue
            yield f"data: {json.dumps({'type': 'message', 'content': chat_message.dict()})}\n\n"

    await stream_task
    yield "data: [DONE]\n\n"

@home_api.post("/stream")
async def stream_agent(): #user_input: StreamInput):
    """
    Stream the agent's response to a user input, including intermediate messages and tokens.

    Use thread_id to persist and continue a multi-turn conversation.
    """
    if "user_id" not in session or not session["user_id"]:
        session["user_id"] = uuid7str()
    if "thread_id" not in session or not session["thread_id"]:
        session["thread_id"] = uuid7str()
    config = RunnableConfig(run_name="RAG Deep Agent /invoke", configurable={"thread_id": session["thread_id"], "user_id":  session["user_id"]})
    logging.info(f"/stream session {session['thread_id']} {session['user_id']}")
    user_input: UserInput = await ProcessCurlInput()
    """
    If it is not curl, then the request must have come from the browser with form data. The following processes it.
    """
    if not user_input or "message" not in user_input:
        form = await request.form
        #logging.debug(f"form: {form}")
        if "prompt" in form and form["prompt"] and len(form["prompt"]):
            user_input = {"message": form["prompt"]}
    if not user_input:
        await flash("Please input your query!", "danger") # https://quart.palletsprojects.com/en/latest/reference/source/quart.helpers.html
        return await Respond("index.html", title="Welcome to LLM-RAG 💬", error="Invalid input!")
    # Expect a single string.
    if isinstance(user_input["message"], (list, tuple)):
        user_input["message"] = user_input["message"][-1]
    @stream_with_context
    async def async_generator():
        message = message_generator(user_input, config)
        yield message.encode()
    return async_generator(), 200
    #return StreamingResponse(message_generator(user_input), media_type="text/event-stream")