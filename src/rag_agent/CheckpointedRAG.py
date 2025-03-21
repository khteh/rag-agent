import asyncio, logging, os, vertexai
from dotenv import load_dotenv
from typing import Annotated, Literal, Sequence
from datetime import datetime
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, MessagesState
from langgraph.graph.graph import (
    END,
    START,
    CompiledGraph,
    Graph,
    Send,
)
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import List, TypedDict
from langgraph.store.memory import InMemoryStore
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
"""
https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/
https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
https://python.langchain.com/docs/tutorials/qa_chat_history/
https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/message.py
https://langchain-ai.github.io/langgraph/how-tos/streaming/#values
"""
load_dotenv()
from .State import State
from ..utils.image import show_graph
#from .State import State
from .VectorStore import vector_store
class CheckpointedRAG():
    _llm = None
    _config = None
    _prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )
    _urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    # Class constructor
    def __init__(self, config: RunnableConfig={}):
        """
        Class CheckpointedRAG Constructor
        """
        self._config = config
        # https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
        self._llm = init_chat_model("gemini-2.0-flash", model_provider="google_vertexai", streaming=True)
        # https://python.langchain.com/docs/integrations/chat/google_vertex_ai_palm/
        """
        self._llm = ChatVertexAI(
                        model="gemini-2.0-flash",
                        temperature=0,
                        max_tokens=None,
                        max_retries=6,
                        stop=None,
                        streaming=True
                    )
        """
        self._llm = self._llm.bind_tools([vector_store.retriever_tool])

    async def Agent(self, state: State, config: RunnableConfig):
        """
        # Step 1: Generate an AIMessage that may include a tool-call to be sent.
        Generate tool call for retrieval or respond

        Invokes the agent model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply end.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with the agent response appended to messages
        """
        logging.info(f"\n=== {self.Agent.__name__} ===")
        logging.debug(f"state: {state}")
        response = await self._llm.ainvoke(state["messages"], config)
        # MessageState appends messages to state instead of overwriting
        return {"messages": [response]}

    async def GradeDocuments(self, state: State) -> Literal["Generate", "Rewrite"]:
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (messages): The current state

        Returns:
            str: A decision for whether the documents are relevant or not
        """
        logging.info(f"\n=== {self.GradeDocuments.__name__} CHECK RELEVANCE ===")
        # Data model
        class grade(BaseModel):
            """Binary score for relevance check."""
            binary_score: str = Field(description="Relevance score 'yes' or 'no'")

        # LLM with tool and validation
        llm_with_tool = self._llm.with_structured_output(grade)

        # Chain
        chain = self._prompt | llm_with_tool

        messages = state["messages"]
        last_message = messages[-1]

        question = messages[0].content
        docs = last_message.content

        scored_result = await chain.ainvoke({"question": question, "context": docs})
        score = scored_result.binary_score

        if score == "yes":
            logging.debug("---DECISION: DOCS RELEVANT---")
            return "Generate"

        else:
            logging.debug(f"---DECISION: DOCS NOT RELEVANT (score: {score})---")
            return "Rewrite"
        
    async def Rewrite(self, state: State, config: RunnableConfig):
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
        msg = [
            HumanMessage(
                content=f""" \n 
                        Look at the input and try to reason about the underlying semantic intent / meaning. \n 
                        Here is the initial question:
                        \n ------- \n
                        {question} 
                        \n ------- \n
                        Formulate an improved question: """,
            )
        ]
        # Grader
        response = await self._llm.ainvoke(msg)
        return {"messages": [response]}

    async def Generate(self, state: State):
        """
        Generate answer

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """
        logging.info(f"\n=== {self.Generate.__name__} ===")
        messages = state["messages"]
        question = messages[0].content
        last_message = messages[-1]

        docs = last_message.content

        # Prompt
        prompt = hub.pull("rlm/rag-prompt")

        # Post-processing
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Chain: https://python.langchain.com/docs/concepts/lcel/
        rag_chain = prompt | self._llm | StrOutputParser()

        # Run
        response = await rag_chain.ainvoke({"context": docs, "question": question})
        return {"messages": [response]}
    
    # Step 3: Generate a response using the retrieved content.
    async def Generate1(self, state: State, config: RunnableConfig):
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
        response = await self._llm.ainvoke(prompt, config)
        #logging.debug(f"\nGenerate() response: {response}")
        return {"messages": [response]}

    async def CreateGraph(self, config: RunnableConfig) -> StateGraph:
        # Compile application and test
        logging.info(f"\n=== {self.CreateGraph.__name__} ===")
        await vector_store.LoadDocuments(self._urls)
        graph_builder = StateGraph(State)
        graph_builder.add_node("Agent", self.Agent)
        graph_builder.add_node("Retrieve", ToolNode([vector_store.retriever_tool])) # Execute the retrieval.
        graph_builder.add_node("Rewrite", self.Rewrite)
        graph_builder.add_node("Generate", self.Generate)
        graph_builder.add_edge(START, "Agent")
        #graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "Agent",
            # Assess agent decision
            tools_condition,
            {
                # Translate the condition outputs to nodes in our graph
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
        return graph_builder.compile(store=InMemoryStore(), checkpointer=MemorySaver(), name="Checkedpoint StateGraph RAG")

async def make_graph(config: RunnableConfig) -> CompiledGraph:
    return await CheckpointedRAG(config).CreateGraph(config)

async def TestDirectResponseWithoutRetrieval(graph, config, message):
    logging.info(f"\n=== {TestDirectResponseWithoutRetrieval.__name__} ===")
    async for step in graph.astream(
        {"messages": [{"role": "user", "content": message}]},
        stream_mode="values",
        config = config
    ):
        step["messages"][-1].pretty_print()

async def Chat(graph, config, messages: List[str]):
    logging.info(f"\n=== {Chat.__name__} ===")
    for message in messages:
        #input_message = "What is Task Decomposition?"
        async for step in graph.astream(
            {"messages": [{"role": "user", "content": message}]},
            stream_mode="values",
            config = config
        ):
            step["messages"][-1].pretty_print()

async def main():
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    config = RunnableConfig(run_name="Checkedpoint StateGraph RAG", thread_id=datetime.now())
    checkpoint_graph = await make_graph(config) # config input parameter is required by langgraph.json to define the graph
    show_graph(checkpoint_graph, "Checkedpoint StateGraph RAG") # This blocks
    """
    graph = checkpoint_graph.get_graph().draw_mermaid_png()
    # Save the PNG data to a file
    with open("/tmp/checkpoint_graph.png", "wb") as f:
        f.write(graph)
    img = Image.open("/tmp/checkpoint_graph.png")
    img.show()
    """
    await TestDirectResponseWithoutRetrieval(checkpoint_graph, config, "Hello, who are you?")
    await Chat(checkpoint_graph, config, ["What is Task Decomposition?", "Can you look up some common ways of doing it?"])

if __name__ == "__main__":
    asyncio.run(main())
