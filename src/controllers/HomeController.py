import re, asyncio, json, jsonpickle, logging, jsonpickle, pickle
from uuid_extensions import uuid7, uuid7str
from typing import AsyncGenerator, Dict, Any, Tuple
from quart import (
    Blueprint,
    flash,
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
from contextlib import asynccontextmanager
from quart.helpers import stream_with_context
from src.common.Authentication import Authentication
from src.models.schema import ChatMessage, UserInput, StreamInput
from langchain_core.callbacks import AsyncCallbackHandler
from langgraph.graph.graph import CompiledGraph
from langchain_core.runnables import RunnableConfig
from urllib.parse import urlparse, parse_qs
from typing_extensions import List, TypedDict
from src.common.ResponseHelper import Respond
from src.common.Response import custom_response
from src.utils.InputParser import parse_input
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

@home_api.post("/invoke")
async def invoke(): #user_input: UserInput) -> ChatMessage:
    """
    Invoke the agent with user input to retrieve a final response.

    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    """
    data = await request.get_data()
    params = parse_qs(data.decode('utf-8'))
    logging.debug(f"data: {data}, params: {params}, type(params): {type(params)}")
    user_input: UserInput = None
    if is_json(data):
        user_input = json.loads(data)
        logging.debug(f"user_input from data: {user_input}")
    elif isinstance(params, dict):
        str_params: str = json.dumps(params)
        logging.debug(f"str_params: {str_params}")
        if str_params and len(str_params) and is_json(str_params):
            user_input = json.loads(str_params)
            logging.debug(f"user_input from params: {user_input}")
    if not user_input:
        await flash("Please input your query!", "danger")
        return await Respond("index.html", title="Welcome to LLM-RAG 💬", error="Invalid input!")
    # Expect a single string.
    if isinstance(user_input['message'], (list, tuple)):
        user_input['message'] = user_input['message'][-1]
    kwargs, run_id = parse_input(user_input)
    logging.debug(f"kwargs: {kwargs}, run_id: {run_id}")
    """
    kwargs: {'input': {'messages': [HumanMessage(content='What is task decomposition?', additional_kwargs={}, response_metadata={})]}, 'config': {'configurable': {'thread_id': '067fa565-02ef-7d89-8000-15cd9d193962'}, 'run_id': UUID('067fa565-02ef-7a08-8000-28b0a62db0e7')}}, run_id: 067fa565-02ef-7a08-8000-28b0a62db0e7
    """
    try:
        result: List[str] = []
        async for step in current_app.agent.astream(
            {"messages": [{"role": "user", "content": user_input['message']}]},
            stream_mode="values", # Use this to stream all values in the state after each step.
            config=kwargs['config'], # This is needed by Checkpointer
        ):
            result.append(step["messages"][-1])
            step["messages"][-1].pretty_print()
        """
        2025-04-12 20:23:19 DEBUG    /invoke respose: content='Task decomposition is a process of breaking down complex tasks or problems into smaller, more manageable steps or subtasks. This technique is used to simplify complicated tasks, making them easier to understand, plan, and execute. 
                    It involves identifying the individual components or steps required to complete a task, and then organizing these steps in a logical order.\n\nTask decomposition can be applied in various contexts, including project management, problem-solving, and decision-making. 
                    It helps individuals or teams to:\n\n1. Clarify complex tasks: By breaking down complex tasks into smaller steps, individuals can better understand what needs to be done.\n2. Identify priorities: Task decomposition helps to identify the most critical steps that need to be completed first.\n3. 
                    Allocate resources: With a clear understanding of the individual steps, resources can be allocated more effectively.\n4. Monitor progress: Decomposing tasks into smaller steps makes it easier to track progress and identify potential bottlenecks.\n\n
                    In the context of artificial intelligence and machine learning, task decomposition is used in techniques such as Chain of Thought (CoT) and Tree of Thoughts. These methods involve breaking down complex tasks into smaller steps, allowing models to utilize more test-time computation and provide insights into their thinking processes.\n\n
                    Overall, task decomposition is a powerful technique for simplifying complex tasks, improving productivity, and enhancing decision-making.' 
        additional_kwargs={} 
        response_metadata={'model': 'llama3.3', 'created_at': '2025-04-12T12:23:19.47198126Z', 'done': True, 'done_reason': 'stop', 'total_duration': 428800185880, 'load_duration': 23447133, 'prompt_eval_count': 674, 'prompt_eval_duration': 19170396674, 'eval_count': 265, 'eval_duration': 409602973053, 'message': 
        Message(role='assistant', content='Task decomposition is a process of breaking down complex tasks or problems into smaller, more manageable steps or subtasks. This technique is used to simplify complicated tasks, making them easier to understand, plan, and execute. It involves identifying the individual components or steps required to complete a task, and then organizing these steps in a logical order.\n\nTask decomposition can be applied in various contexts, including project management, problem-solving, and decision-making. It helps individuals or teams to:\n\n1. Clarify complex tasks: By breaking down complex tasks into smaller steps, individuals can better understand what needs to be done.\n2. Identify priorities: Task decomposition helps to identify the most critical steps that need to be completed first.\n3. Allocate resources: With a clear understanding of the individual steps, resources can be allocated more effectively.\n4. Monitor progress: Decomposing tasks into smaller steps makes it easier to track progress and identify potential bottlenecks.\n\nIn the context of artificial intelligence and machine learning, task decomposition is used in techniques such as Chain of Thought (CoT) and Tree of Thoughts. These methods involve breaking down complex tasks into smaller steps, allowing models to utilize more test-time computation and provide insights into their thinking processes.\n\nOverall, task decomposition is a powerful technique for simplifying complex tasks, improving productivity, and enhancing decision-making.', images=None, tool_calls=None), 'model_name': 'llama3.3'} name='RAG ReAct Agent' id='run-fc59fe44-18ea-44d5-b806-4c8454401703-0' usage_metadata={'input_tokens': 674, 'output_tokens': 265, 'total_tokens': 939} run_id='067fa594-8524-7060-8000-1a4f10981e63'
        """
        ai_message: str = None
        if len(result):
            ai_message = ChatMessage.from_langchain(result[-1])
            #print(f"ai_message: {ai_message}")``
            if ai_message and not ai_message.tool_calls and ai_message.content and len(ai_message.content):
                #output.run_id = str(run_id)
                #return output
                #return await Respond("index.html", title="Welcome to LLM-RAG 💬", greeting=greeting)
                # res.json({ 'message': this.presenter.Message, "errors": this.presenter.Errors });
                logging.debug(f"/invoke respose: {ai_message.content}")
                response = await Respond("index.html", title="Welcome to LLM-RAG 💬", message=ai_message.content)
                logging.debug(f"response: {response}")
                return response
    except Exception as e:
        raise HTTPException(description = str(e))

async def message_generator(user_input: StreamInput) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint.
    """
    kwargs, run_id = parse_input(user_input)

    # Use an asyncio queue to process both messages and tokens in
    # chronological order, so we can easily yield them to the client.
    output_queue = asyncio.Queue(maxsize=10)
    if user_input.stream_tokens:
        kwargs["config"]["callbacks"] = [TokenQueueStreamingHandler(queue=output_queue)]

    # Pass the agent's stream of messages to the queue in a separate task, so
    # we can yield the messages to the client in the main thread.
    async def run_agent_stream():
        async for s in current_app.agent.astream(**kwargs, stream_mode="updates"):
            await output_queue.put(s)
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
                chat_message.run_id = str(run_id)
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

    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.
    """
    data = await request.get_data()
    params = parse_qs(data.decode('utf-8'))
    logging.debug(f"data: {data}, params: {params}")
    user_input: UserInput = jsonpickle.decode(data)
    @stream_with_context
    async def async_generator():
        message = message_generator(user_input)
        yield message.encode()
    return async_generator(), 200
    #return StreamingResponse(message_generator(user_input), media_type="text/event-stream")