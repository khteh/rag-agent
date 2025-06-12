import argparse, asyncio, logging, os, vertexai
from uuid_extensions import uuid7, uuid7str
from typing import Annotated, Literal, Sequence
from datetime import datetime
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, MessagesState
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache
from langgraph.graph.graph import (
    END,
    START,
    Graph,
    Send,
)
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langchain_ollama import OllamaEmbeddings
from psycopg_pool import AsyncConnectionPool, ConnectionPool
from typing_extensions import List, TypedDict
from langgraph.store.memory import InMemoryStore
from langchain_core.prompts import PromptTemplate
from langchain.tools.retriever import create_retriever_tool
from google.api_core.exceptions import ResourceExhausted
from pydantic import BaseModel, Field
"""
https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/
https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
https://python.langchain.com/docs/tutorials/qa_chat_history/
https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/message.py
https://langchain-ai.github.io/langgraph/how-tos/streaming/#values
https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/
"""
from src.config import config as appconfig
from src.common.configuration import Configuration
from src.common.State import State, CustomAgentState
from src.models.DocumentGradeModel import DocumentGradeModel
from src.utils.image import show_graph
from src.Infrastructure.PostgreSQLSetup import PostgreSQLCheckpointerSetup, PostgreSQLStoreSetup
#from .State import State
from src.Infrastructure.VectorStore import VectorStore

class GraphRAG():
    _name: str = "Checkpointed StateGraph RAG"
    _llm = None
    _config = None
    _grading_prompt = PromptTemplate(
            template = "You are a grader assessing relevance of a retrieved document to a user question. \n "
                            "Here is the retrieved document: \n\n {context} \n\n"
                            "Here is the user question: {question} \n"
                            "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
                            "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.",
            input_variables=["context", "question"],
        )
    _rewrite_prompt = PromptTemplate(
        template = "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
                    "Here is the initial question:"
                    "\n ------- \n"
                    "{question}"
                    "\n ------- \n"
                    "Formulate an improved question:",
        input_variables=["question"],
    )
    _generate_prompt = PromptTemplate(
        template = "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer the question. "
                    "If you don't know the answer, just say that you don't know. "
                    "Use three sentences maximum and keep the answer concise.\n"
                    "Question: {question} \n"
                    "Context: {context}",
        input_variables=["context", "question"],
    )
    """
    Prompt
    https://smith.langchain.com/hub
    """
    _rag_prompt = hub.pull("rlm/rag-prompt")
    _urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        "https://mlflow.org/docs/latest/index.html",
        "https://mlflow.org/docs/latest/tracking/autolog.html",
        "https://mlflow.org/docs/latest/getting-started/tracking-server-overview/index.html",
        "https://mlflow.org/docs/latest/python_api/mlflow.deployments.html",        
    ]
    _db_pool: AsyncConnectionPool = None
    _store: AsyncPostgresStore = None
    _vectorStore = None
    _graph: CompiledStateGraph =  None
    _retriever_tool = None
    # Class constructor
    def __init__(self, config: RunnableConfig={}):
        """
        Class GraphRAG Constructor
        """
        self._config = config
        # https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
        # not a vector store but a LangGraph store object. https://github.com/langchain-ai/langchain/issues/30723
        #self._in_memory_store = InMemoryStore(
        #    index={
        #        "embed": OllamaEmbeddings(model=appconfig.EMBEDDING_MODEL, base_url=appconfig.OLLAMA_URI, num_ctx=8192, num_gpu=1, temperature=0),
        #        #"dims": 1536,
        #    }
        #)
        self._db_pool = AsyncConnectionPool(
                conninfo = appconfig.POSTGRESQL_DATABASE_URI,
                max_size = appconfig.DB_MAX_CONNECTIONS,
                kwargs = appconfig.connection_kwargs,
            )
        self._vectorStore = VectorStore(model=appconfig.EMBEDDING_MODEL, chunk_size=1000, chunk_overlap=0)
        # .bind_tools() gives the agent LLM descriptions of each tool from their docstring and input arguments. 
        # If the agent LLM determines that its input requires a tool call, it’ll return a JSON tool message with the name of the tool it wants to use, along with the input arguments.
        # For VertexAI, use VertexAIEmbeddings, model="text-embedding-005"; "gemini-2.0-flash" model_provider="google_genai"
        self._llm = init_chat_model(appconfig.LLM_RAG_MODEL, model_provider="ollama", base_url=appconfig.OLLAMA_URI, streaming=True, temperature=0).bind_tools([self._vectorStore.retriever_tool])
        # https://python.langchain.com/docs/integrations/chat/google_vertex_ai_palm/

    async def Cleanup(self):
        logging.info(f"\n=== {self.Cleanup.__name__} ===")
        await self._store.close()
        await self._db_pool.close()
        #self._in_memory_store.close()

    # https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/
    async def Agent(self, state: CustomAgentState, config: RunnableConfig):
        """
        Invokes the agent model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply end.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with the agent response appended to messages
        """
        logging.info(f"\n=== {self.Agent.__name__} ===")
        logging.debug(f"state: {state}")
        response = await self._llm.with_config(config).ainvoke(state["messages"])
        print(f"{self.Agent.__name__} response: {response.content}")
        # MessageState appends messages to state instead of overwriting
        return {"messages": [response]}

    # https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/
    async def GradeDocuments(self, state: CustomAgentState, config: RunnableConfig) -> Literal["Generate", "Rewrite"]:
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (messages): The current state

        Returns:
            str: A decision for whether the documents are relevant or not

        Note: Edge functions return strings that tell you which node or nodes to navigate to.
        """
        logging.info(f"\n=== {self.GradeDocuments.__name__} CHECK RELEVANCE ===")
        question = state["messages"][0].content
        context = state["messages"][-1].content
        prompt = self._grading_prompt.format(context=context, question=question)
        response = (
            # LLM with tool and validation
            # https://python.langchain.com/docs/how_to/structured_output/
            await self._llm.with_structured_output(DocumentGradeModel).ainvoke(
                [{"role": "user", "content": prompt}]
            )
        )
        score = response.binary_score        
        if score == "yes":
            logging.debug("---DECISION: DOCS RELEVANT---")
            return "Generate"
        else:
            logging.debug(f"---DECISION: DOCS NOT RELEVANT (score: {score})---")
            return "Rewrite"
        
    # https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/
    async def Rewrite(self, state: CustomAgentState, config: RunnableConfig):
        """
        Transform the query to produce a better question.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """
        logging.info(f"\n=== {self.Rewrite.__name__} TRANSFORM QUERY ===")
        messages = state["messages"]
        question = messages[0].content
        prompt = self._rewrite_prompt.format(question=question)
        #print(f"{self.Rewrite.__name__} prompt: {prompt}")
        response = await self._llm.with_config(config).ainvoke([{"role": "user", "content": prompt}])
        print(f"{self.Rewrite.__name__} response: {response.content}")
        logging.debug(f"response: {response.content}")
        return {"messages": [{"role": "user", "content": response.content}]}

    # https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/
    async def Generate(self, state: CustomAgentState, config: RunnableConfig):
        """
        Generate answer

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """
        logging.info(f"\n=== {self.Generate.__name__} ===")
        question = state["messages"][0].content
        context = state["messages"][-1].content
        prompt = self._generate_prompt.format(context=context, question=question)
        #print(f"{self.Generate.__name__} prompt: {prompt}")
        response = await self._llm.ainvoke([{"role": "user", "content": prompt}])
        print(f"{self.Generate.__name__} response: {response}")
        return {"messages": [response]}
    
    # Step 3: Generate a response using the retrieved content.
    async def Generate1(self, state: CustomAgentState, config: RunnableConfig):
        """Generate answer."""
        logging.info(f"\n=== {self.Generate1.__name__} ===")
        #logging.debug(f"\nstate['messages']: {state['messages']}")
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]
        #logging.debug(f"\ntool_messages: {tool_messages}")
        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = f"""
        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer 
        the question. If you don't know the answer, say that you 
        don't know. Use three sentences maximum and keep the 
        answer concise.
        \n\n
        {docs_content}
        """
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages
        #logging.debug(f"\nsystem_message_content: {system_message_content}")
        #logging.debug(f"\nconversation_messages: {conversation_messages}")
        #logging.debug(f"\nprompt: {prompt}")
        # Run
        response = await self._llm.with_config(config).ainvoke(prompt, config)
        #logging.debug(f"\nGenerate() response: {response}")
        return {"messages": [response]}

    async def CreateGraph(self) -> CompiledStateGraph:
        """
        Compile application and test
        https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/
        https://langchain-ai.github.io/langgraph/concepts/low_level/#node-caching
        """
        logging.info(f"\n=== {self.CreateGraph.__name__} ===")
        try:
            cache_policy = CachePolicy(ttl=600) # 10 minutes
            graph_builder = StateGraph(CustomAgentState)
            graph_builder.add_node("Agent", self.Agent, cache_policy = cache_policy)
            graph_builder.add_node("Retrieve", ToolNode([self._vectorStore.retriever_tool]), cache_policy = cache_policy) # Execute the retrieval.
            graph_builder.add_node("Rewrite", self.Rewrite, cache_policy = cache_policy)
            graph_builder.add_node("Generate", self.Generate, cache_policy = cache_policy)
            graph_builder.add_edge(START, "Agent")
            #graph_builder.set_entry_point("query_or_respond")
            # https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph.add_conditional_edges
            graph_builder.add_conditional_edges(
                "Agent",
                # Assess agent decision
                tools_condition,
                {
                    # Translate the condition outputs to nodes in our graph. Which node to go to based on the output of the conditional edge function - tools_condition.
                    "tools": "Retrieve",
                    END: END
                },
            )
            # Edges taken after the `action` node is called.
            graph_builder.add_conditional_edges(
                "Retrieve",
                # Assess agent decision
                self.GradeDocuments,
            )
            graph_builder.add_edge("Generate", END)
            graph_builder.add_edge("Rewrite", "Agent")
            await self._db_pool.open()
            self._store = await PostgreSQLStoreSetup(self._db_pool)
            self._graph = graph_builder.compile(name=self._name, cache=InMemoryCache(), store = self._store)
            #self.ShowGraph(self._graph, self._name) # This blocks
        except ResourceExhausted as e:
            logging.exception(f"google.api_core.exceptions.ResourceExhausted")
        return self._graph

    def ShowGraph(self):
        show_graph(self._agent, self._name) # This blocks

    async def TestDirectResponseWithoutRetrieval(self, config, message: str):
        logging.info(f"\n=== {self.TestDirectResponseWithoutRetrieval.__name__} ===")
        async for step in self._graph.astream(
            {"messages": [{"role": "user", "content": message}]},
            stream_mode="values",
            config = config
        ):
            step["messages"][-1].pretty_print()

    async def Chat(self, config, message: str):
        """
        message is a single string. It can contain multiple questions:
        input_message: str = ("What is task decomposition?\n" "What is the standard method for Task Decomposition?\n" "Once you get the answer, look up common extensions of that method.")
        """
        logging.info(f"\n=== {self.Chat.__name__} ===")
        async with AsyncConnectionPool(
            conninfo = appconfig.POSTGRESQL_DATABASE_URI,
            max_size = appconfig.DB_MAX_CONNECTIONS,
            kwargs = appconfig.connection_kwargs,
        ) as pool:
            # Create the AsyncPostgresSaver
            self._graph.checkpointer = await PostgreSQLCheckpointerSetup(pool)
            async for step in self._graph.astream(
                {"messages": [{"role": "user", "content": message}]},
                stream_mode="values",
                config = config
            ):
                step["messages"][-1].pretty_print()

async def make_graph(config: RunnableConfig) -> CompiledStateGraph:
    return await GraphRAG(config).CreateGraph()

async def main():
    parser = argparse.ArgumentParser(description='Start this LLM-RAG Agent by loading blog content from predefined URLs')
    parser.add_argument('-l', '--load-urls', action='store_true', help='Load documents from URLs')
    args = parser.parse_args()
    #vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    config = RunnableConfig(run_name="Checkpointed StateGraph RAG", thread_id=uuid7str(), user_id=uuid7str())
    graph = GraphRAG(config)
    print(f"args: {args}")
    if args.load_urls:
        await graph.LoadDocuments()
    await graph.CreateGraph() # config input parameter is required by langgraph.json to define the graph
    """
    graph = checkpoint_graph.get_graph().draw_mermaid_png()
    # Save the PNG data to a file
    with open("/tmp/checkpoint_graph.png", "wb") as f:
        f.write(graph)
    img = Image.open("/tmp/checkpoint_graph.png")
    img.show()
    """
    await graph.TestDirectResponseWithoutRetrieval(config, "Hello, who are you?")
    await graph.Chat(config, ("What is task decomposition?\n" "What is the standard method for Task Decomposition?\n" "Once you get the answer, look up common extensions of that method."))

if __name__ == "__main__":
    asyncio.run(main())
