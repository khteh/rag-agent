import argparse, json, asyncio, logging, sys
from uuid_extensions import uuid7, uuid7str
from datetime import datetime
from pathlib import Path
from asyncio import Queue, run, create_task
from typing import Any, Callable, List, AsyncGenerator
from langchain_core.messages.base import BaseMessage, BaseMessageChunk
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, ToolCall
from langchain_core.runnables import RunnableConfig, ensure_config
from langgraph.graph.state import CompiledStateGraph
from langchain.agents import create_agent
from langgraph.store.base import BaseStore
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.callbacks import AsyncCallbackHandler
from psycopg_pool import AsyncConnectionPool
from langchain_ollama import OllamaEmbeddings
from deepagents import create_deep_agent, CompiledSubAgent
# https://github.com/langchain-ai/deepagents
#https://python.langchain.com/docs/tutorials/qa_chat_history/
#https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
#https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
#https://langchain-ai.github.io/langgraph/how-tos/streaming/#values
#https://python.langchain.com/docs/how_to/configure/
#https://langchain-ai.github.io/langgraph/how-tos/
from src.config import config as appconfig
from src.rag_agent.RAGPrompts import RAG_INSTRUCTIONS, SUBAGENT_DELEGATION_INSTRUCTIONS, RAG_WORKFLOW_INSTRUCTIONS
from src.models.schema import ChatMessage, UserInput, StreamInput
from src.rag_agent.Tools import upsert_memory, think_tool
from src.Infrastructure.VectorStore import VectorStore
from src.common.State import CustomAgentState
from src.utils.image import show_graph
from src.Healthcare.RAGAgent import RAGAgent as HealthAgent
from src.Healthcare.prompts import HEALTHCARE_INSTRUCTIONS
from src.rag_agent.Tools import current_timestamp, ground_search, upsert_memory
from src.common.Configuration import Configuration
from src.Infrastructure.Backend import composite_backend
from src.Infrastructure.PostgreSQLSetup import PostgreSQLCheckpointerSetup, PostgreSQLStoreSetup

class TokenQueueStreamingHandler(AsyncCallbackHandler):
    """LangChain callback handler for streaming LLM tokens to an asyncio queue."""
    def __init__(self, queue: Queue):
        self.queue = queue

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        if token:
            await self.queue.put(token)

class RAGAgent():
    _name:str = "RAG Deep Agent"
    _llm = None
    _urls = [
        {"url": "https://lilianweng.github.io/", "type": "article"},
        {"url": "https://lilianweng.github.io/posts/2023-06-23-agent/", "type": "article", "filter": ("post-content", "post-title", "post-header")},
        {"url": "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/", "type": "article", "filter": ("post-content", "post-title", "post-header")},
        {"url": "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/", "type": "article", "filter": ("post-content", "post-title", "post-header")},
        {"url": "https://mlflow.org/docs/latest/ml/", "type": "article", "filter": ("theme-doc-markdown markdown")},
        {"url": "https://mlflow.org/docs/latest/ml/tracking/autolog/", "type": "article", "filter": ("theme-doc-markdown markdown")},
        {"url": "https://mlflow.org/docs/latest/ml/tracking/", "type": "article", "filter": ("theme-doc-markdown markdown")},
        {"url": "https://mlflow.org/docs/latest/python_api/mlflow.deployments.html", "type": "class", "filter": ("section")}
    ]
    # Limits
    _max_concurrent_research_units = 3
    _max_researcher_iterations = 3
    # Get current date
    # Combine orchestrator instructions (RESEARCHER_INSTRUCTIONS only for sub-agents)
    _INSTRUCTIONS = (
        RAG_WORKFLOW_INSTRUCTIONS
        + "\n\n"
        + "=" * 80
        + "\n\n"
        + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
            max_concurrent_research_units = _max_concurrent_research_units,
            max_researcher_iterations = _max_researcher_iterations,
        )
    )
    _db_pool: AsyncConnectionPool = None
    _store: AsyncPostgresStore = None
    _checkpointer = None
    _vectorStore = None
    _tools: List[Callable[..., Any]] = None
    _healthcare_rag: HealthAgent = None
    _healthcare_agent = None
    _ragagent = None
    _subagents = None
    _healthcare_subagent: CompiledSubAgent = None
    _rag_subagent: CompiledSubAgent = None
    _agent = None
    # Class constructor
    def __init__(self, db_pool:AsyncConnectionPool = None, store: AsyncPostgresStore = None, checkpointer: AsyncPostgresSaver = None):
        """
        Class RAGAgent Constructor
        """
        #atexit.register(self.Cleanup)
        #vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
        # .bind_tools() gives the agent LLM descriptions of each tool from their docstring and input arguments. 
        # If the agent LLM determines that its input requires a tool call, it’ll return a JSON tool message with the name of the tool it wants to use, along with the input arguments.
        # For VertexAI, use VertexAIEmbeddings, model="text-embedding-005"; "gemini-2.0-flash" model_provider="google_genai"
        # not a vector store but a LangGraph store object. https://github.com/langchain-ai/langchain/issues/30723
        self._db_pool = db_pool or AsyncConnectionPool(
                        conninfo = appconfig.POSTGRESQL_DATABASE_URI,
                        max_size = appconfig.DB_MAX_CONNECTIONS,
                        kwargs = appconfig.connection_kwargs,
                        open = False
                    )
        self._store = store
        self._checkpointer = checkpointer
        self._vectorStore = VectorStore(model=appconfig.EMBEDDING_MODEL, chunk_size=1000, chunk_overlap=0)
        self._healthcare_rag = HealthAgent(self._db_pool, self._store, self._checkpointer)
        # GoogleSearch ground_search works well but it will sometimes take precedence and overwrite the ingested data into Chhroma and Neo4J. So, exclude it for now until it is really needed.
        # Use it as a custom subagent
        self._tools = [self._vectorStore.retriever_tool, upsert_memory, think_tool]
        if appconfig.BASE_URI:
            self._llm = init_chat_model(appconfig.LLM_RAG_MODEL, model_provider=appconfig.MODEL_PROVIDER, base_url=appconfig.BASE_URI, streaming=True, temperature=0)
        else:
            self._llm = init_chat_model(appconfig.LLM_RAG_MODEL, model_provider=appconfig.MODEL_PROVIDER, streaming=True, temperature=0)

    #def Cleanup(self):
    #    https://github.com/minrk/asyncio-atexit/issues/11
    #    logging.info(f"\n=== {self.Cleanup.__name__} ===")
    #    self._store.close()
    #    self._db_pool.close()
        #self._in_memory_store.close()
    async def prepare_model_inputs(self, state: CustomAgentState, config: RunnableConfig, store: BaseStore):
        # Retrieve user memories and add them to the system message
        # This function is called **every time** the model is prompted. It converts the state to a prompt
        config = ensure_config(config)
        user_id = config.get("configurable", {}).get("user_id")
        namespace = ("memories", user_id)
        memories = [m.value["data"] for m in await store.asearch(namespace)]
        system_msg = f"User memories: {', '.join(memories)}"
        return [{"role": "system", "content": system_msg}] + state["messages"]

    async def LoadDocuments(self):
        logging.info(f"\n=== {self.LoadDocuments.__name__} ===")
        await self._vectorStore.LoadDocuments(self._urls)

    async def CreateGraph(self) -> CompiledStateGraph:
        logging.info(f"\n=== {self.CreateGraph.__name__} ===")
        try:
            # https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent
            # https://github.com/langchain-ai/langchain/issues/30723
            # https://langchain-ai.github.io/langgraph/how-tos/cross-thread-persistence/
            # https://github.com/langchain-ai/langgraph/blob/main/libs/prebuilt/langgraph/prebuilt/chat_agent_executor.py#L241
            await self._db_pool.open()
            if self._store is None:
                self._store = AsyncPostgresStore(self._db_pool, index={
                            "embed": OllamaEmbeddings(model=appconfig.EMBEDDING_MODEL, base_url=appconfig.BASE_URI, num_ctx=8192, num_gpu=1, temperature=0),
                            "dims": appconfig.EMBEDDING_DIMENSIONS, # Note: Every time when this value changes, remove the store<foo> tables in the DB so that store.setup() runs to recreate them with the right dimensions.
                        }
                )
                await PostgreSQLStoreSetup(self._db_pool, self._store) # store is needed when creating the ReAct agent / StateGraph for InjectedStore to work
            if self._checkpointer is None:
                self._checkpointer = AsyncPostgresSaver(self._db_pool)
                await PostgreSQLCheckpointerSetup(self._db_pool, self._checkpointer)
            self._ragagent = create_agent(self._llm, self._tools, context_schema = Configuration, state_schema = CustomAgentState, name = self._name, system_prompt = RAG_INSTRUCTIONS, store = self._store, checkpointer = self._checkpointer)
            # Use it as a custom subagent
            self._rag_subagent = CompiledSubAgent(
                name="RAG Sub-Agent",
                description="Specialized agent which answers users' questions based on the information in the vector store",
                system_prompt = RAG_INSTRUCTIONS, # developer provided instructions
                runnable=self._ragagent
            )            
            self._healthcare_agent = await self._healthcare_rag.CreateGraph()
            self._healthcare_subagent = CompiledSubAgent(
                name="Healthcare Sub-Agent",
                description= "Specialized healthcare AI assistant",
                system_prompt = HEALTHCARE_INSTRUCTIONS, # developer provided instructions
                runnable= self._healthcare_agent
            )
            self._subagents = [self._healthcare_subagent, self._rag_subagent]
            self._agent = create_deep_agent(
                model = self._llm,
                tools = [current_timestamp, upsert_memory, think_tool], # [ground_search]
                backend = composite_backend,
                checkpointer = self._checkpointer, # https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/state.py#L828
                store = self._store,
                system_prompt = self._INSTRUCTIONS, # developer provided instructions
                subagents = self._subagents
            )
            # self.ShowGraph() # This blocks
        except Exception as e:
            logging.exception(f"Exception! {e}")
        return self._agent

    def ShowGraph(self):
        show_graph(self._agent, self._name) # This blocks

    async def ShowThreads(self):
        # https://docs.langchain.com/oss/python/langgraph/add-memory
        # AsyncPostgresSaver uses a connection pool
        if self._checkpointer is None:
            self._checkpointer = await PostgreSQLCheckpointerSetup(self._db_pool)
        thread_ids = await self._GetAllThreadIds()
        for id in thread_ids:
            config = {
                "configurable": {
                    "thread_id": id
                }
            }
            # Use 'async for' to iterate over the asynchronous iterator
            async for checkpoint_tuple in self._checkpointer.alist(config): #, limit=10):
                # Accessing the checkpoint data and metadata
                # Checkpoint structure is complex, often needs specific parsing
                # Example of accessing a messages channel (structure may vary)
                messages = checkpoint_tuple.checkpoint.get("channel_values", {}).get("messages", [])
                print(f"{id}: {len(messages)} messages")
                logging.debug(f"{id}: {len(messages)} messages")
        await self._db_pool.close() # Close pool when done (optional, depending on app lifecycle)
        return thread_ids
    
    async def DeleteAllCheckpointsForThread(self, id:int):
        logging.info(f"\n=== {self.DeleteAllCheckpointsForThread.__name__} ===")
        if self._checkpointer is None:
            self._checkpointer = await PostgreSQLCheckpointerSetup(self._db_pool)
        await self._checkpointer.adelete_thread(id)

    async def DeleteAllThreads(self):
        logging.info(f"\n=== {self.DeleteAllThreads.__name__} ===")
        if self._checkpointer is None:
            self._checkpointer = await PostgreSQLCheckpointerSetup(self._db_pool)
        thread_ids = await self._GetAllThreadIds()
        for id in thread_ids:
            await self.DeleteAllCheckpointsForThread(id)

    async def ChatAgent(self, config: RunnableConfig, message: str, modes: List[str] = ["values"], subgraphs: bool = False, output_queue: Queue = None) -> str:
        """
        message is a single string. It can contain multiple questions:
        input_message: str = ("What is task decomposition?\n" "What is the standard method for Task Decomposition?\n" "Once you get the answer, look up common extensions of that method.")
        """
        logging.info(f"\n=== {self.ChatAgent.__name__} modes: {modes}, subgraphs: {subgraphs} ===")
        result: str = "Oops, there was some error. Please try again!"
        for attempt in range(3):
            try:
                # https://langchain-ai.github.io/langgraph/concepts/streaming/
                # https://langchain-ai.github.io/langgraph/how-tos/#streaming
                messages: List[BaseMessage] = []
                async for stream_mode, data in self._agent.astream( # The output of each step in the graph. The output shape depends on the stream_mode.
                    {"messages": [{"role": "user", "content": message}]},
                    stream_mode = modes, # Use this to stream all values in the state after each step.
                    config = config, # This is needed by Checkpointer
                    subgraphs = subgraphs
                ):
                    logging.debug(f"stream_mode: {stream_mode}")
                    if stream_mode == "values":
                        #logging.debug(f"data type: {type(data["messages"][-1])}")
                        messages.append(data["messages"][-1])
                        data["messages"][-1].pretty_print()
                    elif stream_mode == "updates":
                        for source, update in data.items():
                            if source in ("model", "tools"):
                                #logging.debug(f"update type: {type(update["messages"][-1])}")
                                logging.debug(f"source: {source}")
                                update["messages"][-1].pretty_print()
                                await output_queue.put(update["messages"][-1])
                #2025-04-12 20:23:19 DEBUG    /invoke respose: content='Task decomposition is a process of breaking down complex tasks or problems into smaller, more manageable steps or subtasks. This technique is used to simplify complicated tasks, making them easier to understand, plan, and execute. 
                #            It involves identifying the individual components or steps required to complete a task, and then organizing these steps in a logical order.\n\nTask decomposition can be applied in various contexts, including project management, problem-solving, and decision-making. 
                #            It helps individuals or teams to:\n\n1. Clarify complex tasks: By breaking down complex tasks into smaller steps, individuals can better understand what needs to be done.\n2. Identify priorities: Task decomposition helps to identify the most critical steps that need to be completed first.\n3. 
                #            Allocate resources: With a clear understanding of the individual steps, resources can be allocated more effectively.\n4. Monitor progress: Decomposing tasks into smaller steps makes it easier to track progress and identify potential bottlenecks.\n\n
                #            In the context of artificial intelligence and machine learning, task decomposition is used in techniques such as Chain of Thought (CoT) and Tree of Thoughts. These methods involve breaking down complex tasks into smaller steps, allowing models to utilize more test-time computation and provide insights into their thinking processes.\n\n
                #            Overall, task decomposition is a powerful technique for simplifying complex tasks, improving productivity, and enhancing decision-making.' 
                #additional_kwargs={}
                #response_metadata={'model': 'llama3.3', 'created_at': '2025-04-12T12:23:19.47198126Z', 'done': True, 'done_reason': 'stop', 'total_duration': 428800185880, 'load_duration': 23447133, 'prompt_eval_count': 674, 'prompt_eval_duration': 19170396674, 'eval_count': 265, 'eval_duration': 409602973053, 'message': 
                #Message(role='assistant', content='Task decomposition is a process of breaking down complex tasks or problems into smaller, more manageable steps or subtasks. This technique is used to simplify complicated tasks, making them easier to understand, plan, and execute. It involves identifying the individual components or steps required to complete a task, and then organizing these steps in a logical order.\n\nTask decomposition can be applied in various contexts, including project management, problem-solving, and decision-making. It helps individuals or teams to:\n\n1. Clarify complex tasks: By breaking down complex tasks into smaller steps, individuals can better understand what needs to be done.\n2. Identify priorities: Task decomposition helps to identify the most critical steps that need to be completed first.\n3. Allocate resources: With a clear understanding of the individual steps, resources can be allocated more effectively.\n4. Monitor progress: Decomposing tasks into smaller steps makes it easier to track progress and identify potential bottlenecks.\n\nIn the context of artificial intelligence and machine learning, task decomposition is used in techniques such as Chain of Thought (CoT) and Tree of Thoughts. These methods involve breaking down complex tasks into smaller steps, allowing models to utilize more test-time computation and provide insights into their thinking processes.\n\nOverall, task decomposition is a powerful technique for simplifying complex tasks, improving productivity, and enhancing decision-making.', images=None, tool_calls=None), 'model_name': 'llama3.3'} name='RAG ReAct Agent' id='run-fc59fe44-18ea-44d5-b806-4c8454401703-0' usage_metadata={'input_tokens': 674, 'output_tokens': 265, 'total_tokens': 939}
                ai_message: ChatMessage = None
                if len(messages):
                    ai_message = ChatMessage.from_langchain(messages[-1])
                    if ai_message and not ai_message.tool_calls and ai_message.content and len(ai_message.content):
                        logging.debug(f"respose: {ai_message.content}")
                        #logging.debug(f"response: {response}")
                        return True, ai_message.content
            except Exception as e:
                # https://langchain-ai.github.io/langgraph/troubleshooting/errors/INVALID_CHAT_HISTORY/
                logging.exception(f"ChatAgent exception! {str(e)}, repr: {repr(e)}")
                if attempt == 2:
                    return False, result
                if "Found AIMessages with tool_calls that do not have a corresponding ToolMessage" in str(e):
                    #state = await current_app.agent.with_config(config).aget_state() XXX: This doesn't work! Why!?!
                    state = await self._agent.aget_state({"configurable": {"thread_id": config.thread_id, "user_id": config.user_id}})
                    tool_messages = set()
                    outstanding_tool_call_ids = set()
                    # Assumption: state.messages is chronologically ordered
                    for message in reversed(state.values["messages"]):
                        msg = ChatMessage.from_langchain(message)
                        if msg.type == "tool":
                            tool_messages.add(msg.tool_call_id)
                        elif msg.type == "ai" and len(msg.tool_calls):
                            for tc in msg.tool_calls:
                                if tc["id"] not in tool_messages:
                                    outstanding_tool_call_ids.add(tc["id"])
                    for id in outstanding_tool_call_ids:
                        # provide ToolMessages that match existing tool calls and call graph.invoke({'messages': [ToolMessage(...)]}). 
                        # NOTE: this will append the messages to the history and run the graph from the START node.
                        await self._agent.with_config(config).ainvoke({'messages': [ToolMessage(content="I don't know!", tool_call_id=id)]})
        return False, result

    async def message_generator(self, user_input: StreamInput, config: RunnableConfig) -> AsyncGenerator[str, None]:
        """
        Generate a stream of messages from the agent.
        This is the workhorse method for the /stream endpoint.
        """
        logging.info(f"\n=== {self.message_generator.__name__} ===")
        # Use an asyncio queue to process both messages and tokens in
        # chronological order, so we can easily yield them to the client.
        output_queue = Queue(maxsize=10)
        if user_input.stream_tokens:
            config["callbacks"] = [TokenQueueStreamingHandler(queue=output_queue)]

        stream_task = create_task(self._run_agent_stream(config, user_input.message, ["updates"], True, output_queue))
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
                yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"
        await stream_task
        yield "data: [DONE]\n\n"

    # Pass the agent's stream of messages to the queue in a separate task, so
    # we can yield the messages to the client in the main thread.
    async def _run_agent_stream(self, config: RunnableConfig, message: str, modes: List[str], subgraphs:bool, output_queue: Queue):
        # https://docs.langchain.com/oss/python/langchain/streaming/overview#streaming-from-sub-agents
        logging.info(f"\n=== run_agent_stream ===")
        await self.ChatAgent(config, message, modes, subgraphs, output_queue)
        await output_queue.put(None)

    async def _GetAllThreadIds(self):
        logging.info(f"\n=== {self._GetAllThreadIds.__name__} ===")
        async with self._db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT DISTINCT thread_id FROM checkpoints")
                thread_ids = [row[0] for row in await cur.fetchall()]
        logging.debug(f"{len(thread_ids)} threads")
        return thread_ids

async def make_graph() -> CompiledStateGraph:
    """
    This should only be used by the langgraph cli as defined in langgraph.json
    """
    return await RAGAgent().CreateGraph()

async def main():
    """
    https://docs.python.org/3/library/argparse.html
    'store_true' and 'store_false' - These are special cases of 'store_const' used for storing the values True and False respectively. In addition, they create default values of False and True respectively:
    """
    parser = argparse.ArgumentParser(description='LLM-RAG deep agent answering user questions about healthcare system and AI/ML')
    parser.add_argument('-l', '--load-urls', action='store_true', help='Load documents from URLs')
    parser.add_argument('-t', '--show-threads', action='store_true', help='Show history of all threads')
    parser.add_argument('-d', '--delete-threads', action='store_true', help='Delete history of all threads')
    parser.add_argument('-a', '--ai-ml', action='store_true', help="Ask questions regarding AI/ML which should be answered based on Lilian's blog")
    parser.add_argument('-m', '--mlflow', action='store_true', help="Ask questions regarding MLFlow")
    parser.add_argument('-n', '--neo4j-graph', action='store_true', help='Ask question with answer in Neo4J graph database store')
    parser.add_argument('-u', '--stream-updates', action='store_true', help='Stream updates instead of theh complete message')
    parser.add_argument('-v', '--neo4j-vector', action='store_true', help='Ask question with answers in Neo4J vector store')
    parser.add_argument('-b', '--neo4j', action='store_true', help='Ask question with answers in both Neo4J vector and graph stores')
    parser.add_argument('-w', '--wait-time', action='store_true', help='Ask hospital waiting time using answer from mock API endpoint')
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    Path("output/user_questions.md").unlink(missing_ok=True)
    Path("output/final_answer.md").unlink(missing_ok=True)
    # httpx library is a dependency of LangGraph and is used under the hood to communicate with the AI models.
    #vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    rag = RAGAgent()
    await rag.CreateGraph()
    """
    graph = agent.get_graph().draw_mermaid_png()
    # Save the PNG data to a file
    with open("/tmp/agent_graph.png", "wb") as f:
        f.write(graph)
    img = Image.open("/tmp/agent_graph.png")
    img.show()
    """
    print(f"args: {args}")
    if args.load_urls:
        await rag.LoadDocuments()
        return
    elif args.show_threads:
        await rag.ShowThreads()
        return
    elif args.delete_threads:
        await rag.DeleteAllThreads()
        return
    input_message: str = ""
    if args.ai_ml:
        input_message = ("What is task decomposition?\n" "What is the standard method for Task Decomposition?\n" "Once you get the answer, look up common extensions of that method.")
    elif args.mlflow:
        input_message = "What is MLFLow?"
    elif args.wait_time:
        input_message = "Which hospital has the shortest wait time?"
    elif args.neo4j_graph:
        input_message = "Which physician has treated the most patients covered by Cigna?"
    elif args.neo4j_vector:
        input_message = "What have patients said about hospital efficiency? Mention details from specific reviews."
    elif args.neo4j:
        input_message = "Query the graph database to show me the reviews written by patient 7674"
    config = RunnableConfig(run_name="RAG Deep Agent", configurable={"thread_id": uuid7str(), "user_id": uuid7str()})
    if args.stream_updates:
        input = StreamInput(message = input_message, thread_id = config["configurable"]["thread_id"], stream_tokens = True)
        async for answer in rag.message_generator(input, config):
            print(answer)
    else:
        await rag.ChatAgent(config, input_message)

if __name__ == "__main__":
    run(main())
