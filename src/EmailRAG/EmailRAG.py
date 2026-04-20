import atheris, argparse
with atheris.instrument_imports():
    import asyncio, logging, sys
    from datetime import datetime
    from pathlib import Path
    from uuid_extensions import uuid7, uuid7str
    from google.api_core.exceptions import ResourceExhausted
    from langchain.chat_models import init_chat_model
    from langchain_core.runnables import RunnableConfig, ensure_config
    from langgraph.graph import StateGraph, MessagesState
    from langgraph.types import CachePolicy
    from langgraph.cache.memory import InMemoryCache
    from langgraph.graph import (
        END,
        START,
    )
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.store.postgres.aio import AsyncPostgresStore
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg_pool import AsyncConnectionPool, ConnectionPool
    from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate, SystemMessagePromptTemplate
    from typing_extensions import List, TypedDict
    from deepagents import create_deep_agent, CompiledSubAgent
    from langchain_ollama import OllamaEmbeddings
    """
    https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/
    https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
    https://python.langchain.com/docs/tutorials/qa_chat_history/
    https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
    https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/message.py
    https://langchain-ai.github.io/langgraph/how-tos/streaming/#values
    https://python.langchain.com/docs/how_to/configure/
    """
    from src.EmailRAG.EmailPrompts import EMAIL_PARSER_INSTRUCTIONS, EMAIL_PROCESSING_INSTRUCTIONS
    from src.EmailRAG.RobustEmailModelParser import RobustEmailModelParser
    from src.config import config as appconfig
    from src.common.State import EmailRAGState
    from src.common.ContextSchema import ContextSchema
    from src.utils.image import show_graph
    from src.models import ChatMessage
    from src.models.EmailModel import EmailModel
    from src.models.EscalationModel import EscalationCheckModel
    from src.Infrastructure.VectorStore import VectorStore
    from src.Infrastructure.PostgreSQLSetup import PostgreSQLCheckpointerSetup, PostgreSQLStoreSetup
    from src.Infrastructure.Backend import composite_backend
    from src.rag_agent.Tools import upsert_memory, think_tool, RAGMemoryManager, RAGMemorySearcher
    from data.sample_emails import EMAILS
# https://realpython.com/langgraph-python/

class EmailRAG():
    _graphName :str = "Email RAG StateGraph"
    _agentName :str = "Email RAG Agent"
    _llm = None
    _chainLLM = None
    _config = None
    _in_thinking = False
    _email_model_parser: RobustEmailModelParser = None
    _email_parser_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an expert email parser.
                Extract date from the Date: field, name and email from the From: field, project id from the Subject: field or email body text, 
                phone number, site location, violation type, required changes, compliance deadline, and maximum potential fine from the email body text.
                If any of the fields aren't present, don't populate them. Don't populate fields if they're not present in the email.
                Try to cast dates into the dd-mm-YYYY format. Ignore the timestamp and timezone part of the Date.
                IMPORTANT: All fields must be plain scalar values (string, number, or null). Never use nested objects, dicts, or arrays for any field.
                For site_location, produce a single address string (e.g. "456 Sunset Boulevard, Los Angeles, CA").
                For violation_types, return each distinct violation as a separate list entry.
                For required_changes, return each distinct corrective action as a separate list entry.

                Here's the email:
                {email}
                """,
            ),
            ("human", "{email}"),
        ]
    )
    _escalation_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Determine whether the following email received from a regulatory body requires immediate escalation.
                Immediate escalation is required when {escalation_criteria}.
                Populate escalation_reason with a concise explanation referencing the specific criteria met.
                If escalation is warranted, set escalation_priority to one of:
                  - 'immediate' — safety risk or stop-work order threat
                  - 'urgent'    — large potential fine or tight compliance deadline
                  - 'standard'  — other criteria met
                Leave escalation_priority null and explain in escalation_reason when escalation is not required.
                You MUST respond with a single valid JSON object containing exactly these three keys:
                  "needs_escalation" (boolean), "escalation_reason" (string or null), "escalation_priority" (string or null).
                Do NOT rename or alias these keys. Do NOT use YAML, markdown, bullet points,
                bold text, headings, or any formatting outside the JSON.

                Here's the email:
                {email}
                """,
            ),
            ("human", "{email}"),
        ]
    )
    # https://realpython.com/build-llm-rag-chatbot-with-langchain/#chains-and-langchain-expression-language-lcel
    _db_pool: AsyncConnectionPool = None
    _store: AsyncPostgresStore = None
    _checkpointer = None
    _vectorStore = None
    _email_parser_chain = None
    _escalation_chain = None
    _parser_graph: CompiledStateGraph = None
    _email_graph: CompiledStateGraph = None
    _parser_subagent: CompiledSubAgent = None
    _subagents = None
    _agent: CompiledStateGraph = None
    # Class constructor
    def __init__(self, config: RunnableConfig={}):
        """
        Class EmailRAG Constructor
        """
        self._config = config
        # https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
        # https://python.langchain.com/docs/integrations/chat/google_vertex_ai_palm/
        # https://python.langchain.com/docs/how_to/structured_output/
        # .bind_tools() gives the agent LLM descriptions of each tool from their docstring and input arguments. 
        # If the agent LLM determines that its input requires a tool call, it’ll return a JSON tool message with the name of the tool it wants to use, along with the input arguments.
        # For VertexAI, use VertexAIEmbeddings, model="text-embedding-005"; "gemini-2.0-flash" model_provider="google_genai"
        # not a vector store but a LangGraph store object. https://github.com/langchain-ai/langchain/issues/30723
        #self._in_memory_store = InMemoryStore(
        #    index={
        #        "embed": OllamaEmbeddings(model=appconfig.EMBEDDING_MODEL, base_url=appconfig.OLLAMA_CLOUD_URI, num_ctx=8192, num_gpu=1, temperature=0),
        #        #"dims": 1536,
        #    }
        #)
        #asyncio_atexit.register(self.Cleanup)
        self._db_pool = AsyncConnectionPool(
                conninfo = appconfig.POSTGRESQL_DATABASE_URI,
                max_size = appconfig.DB_MAX_CONNECTIONS,
                kwargs = appconfig.connection_kwargs,
                open = False
            )
        self._email_model_parser = RobustEmailModelParser()
        self._vectorStore = VectorStore(model=appconfig.EMBEDDING_MODEL, chunk_size=1000, chunk_overlap=0)
        if appconfig.OLLAMA_CLOUD_URI:
            self._llm = init_chat_model(appconfig.LLM_RAG_MODEL, model_provider=appconfig.MODEL_PROVIDER, base_url=appconfig.OLLAMA_CLOUD_URI, api_key=appconfig.OLLAMA_API_KEY, streaming=True, temperature=0, think="high")
            self._chainLLM = init_chat_model(appconfig.LLM_RAG_MODEL, model_provider=appconfig.MODEL_PROVIDER, base_url=appconfig.OLLAMA_CLOUD_URI, api_key=appconfig.OLLAMA_API_KEY, temperature=0, think="high")
        else:
            self._llm = init_chat_model(appconfig.LLM_RAG_MODEL, model_provider=appconfig.MODEL_PROVIDER, api_key=appconfig.OLLAMA_API_KEY, streaming=True, temperature=0, think="high")
            self._chainLLM = init_chat_model(appconfig.LLM_RAG_MODEL, model_provider=appconfig.MODEL_PROVIDER, api_key=appconfig.OLLAMA_API_KEY, temperature=0, think="high")
        self._email_parser_chain = (
            self._email_parser_prompt | self._chainLLM | self._email_model_parser
            #| self._chainLLM.with_structured_output(EmailModel, method="json_schema")
        )
        self._escalation_chain = (
            self._escalation_prompt
            | self._chainLLM.with_structured_output(EscalationCheckModel, method="json_schema")
        )
    #async def Cleanup(self):
    #    https://github.com/minrk/asyncio-atexit/issues/11
    #    logging.info(f"\n=== {self.Cleanup.__name__} ===")
    #    await self._db_pool.close()
    async def ParseEmail(self, state: EmailRAGState, config: RunnableConfig) -> EmailRAGState:
        """
        Extract structured fields from a regulatory notice.
        This should be used when the email message comes from
        a regulatory body or auditor regarding a property or
        construction site that the company works on.
        https://github.com/langchain-ai/deepagents/issues/613
        """
        logging.info(f"\n=== {self.ParseEmail.__name__} ===")
        config = ensure_config(config)
        timestamp = config.get("configurable", {}).get("timestamp")
        with open(f"output/email_request_{timestamp}.md", 'r') as f:
            email:str = ""
            for l in f:
                if "escalation dollar criteria" in l.lower():
                    try:
                        state["escalation_dollar_criteria"] = float(
                            l.split(":", 1)[1].strip()
                        )
                    except ValueError as e:
                        logging.critical(f"{self.ParseEmail.__name__} ValueError: {e}")
                elif "escalation criteria" in l.lower():
                    state["escalation_text_criteria"] = l.split(":", 1)[1].strip()
                elif "escalation emails" in l.lower():
                    state["escalation_emails"] = [
                        e.strip() for e in l.split(":", 1)[1].strip().split(",")
                    ]
                elif l and l.strip():
                    email += l
        state["email"] = email.strip()
        logging.debug(f"state: {state}")
        #logging.debug(f"config: {config}")
        state["extract"] = await self._email_parser_chain.with_config(config).ainvoke({"email": state["email"]})
        logging.debug(f"Extract: {state["extract"]}")
        return state

    async def NeedsEscalation(self, state: EmailRAGState, config: RunnableConfig) -> EmailRAGState:
        """
        Determine if an email needs escalation
        """
        logging.info(f"\n=== {self.NeedsEscalation.__name__} ===")
        assert self._escalation_chain
        #logging.debug(f"config: {config}")
        logging.debug(f"state: {state}")
        result: EscalationCheckModel = await self._escalation_chain.with_config(config).ainvoke({"email": state["email"], "escalation_criteria": state["escalation_text_criteria"]})
        logging.debug(f"result: {result}")
        extract = state.get("extract")
        dollar_threshold = state.get("escalation_dollar_criteria")
        fine_exceeds_threshold = (
            extract is not None
            and extract.max_potential_fine is not None
            and dollar_threshold is not None
            and extract.max_potential_fine >= dollar_threshold
        )

        state["escalate"] = result.needs_escalation or fine_exceeds_threshold

        reason = result.escalation_reason
        if fine_exceeds_threshold and not result.needs_escalation:
            fine_note = (
                f"Fine of {extract.max_potential_fine} exceeds the "  # type: ignore[union-attr]
                f"{dollar_threshold} threshold."
            )
            reason = f"{reason}; {fine_note}" if reason else fine_note

        state["escalation_reason"] = reason
        state["escalation_priority"] = result.escalation_priority

        logging.debug(f"state: {state}")
        return state
    
    async def ParseEmailSummarizer(self, state: EmailRAGState, config: RunnableConfig) -> EmailRAGState:
        """
        Provides result summary of this subagent in the following format
        {
            'date_str': "The date of the email reformatted to match dd-mm-YYYY. This is usually found in the Date: field in the email. Ignore the timestamp and timezone part of the Date:",
            'name': "The name of the email sender. This is usually found in the From: field in the email formatted as name <email>",
            'phone': "The phone number of the email sender (if present in the message). This is usually found in the signature at the end of the email body.",
            'email': "The email addreess of the email sender (if present in the message). This is usually found in the From: field in the email formatted as name <email>",
            'project_id': "The project ID (if present in the message) - must be an integer. This is usually found in the Subject: field or email body text",
            'site_location': "The site location of the project (if present in the message). Use the full address if possible.",
            'violation_types': "The type of violation (if present in the message)",
            'required_changes': "The required changes specified by the email (if present in the message)",
            'compliance_deadline_str': "The date that the company must comply (if any) reformatted to match dd-mm-YYYY",
            'max_potential_fine': "The maximum potential fine (if any) - must be a float."
        }
        """
        logging.info(f"\n=== {self.ParseEmailSummarizer.__name__} ===")
        config = ensure_config(config)
        timestamp = config.get("configurable", {}).get("timestamp")
        extract = state.get("extract", "{}")
        messages = extract.model_dump_json(indent=4)
        if state.get("escalate"):
            messages += "\nThis email warrants an escalation."
            if state.get("escalation_priority"):
                messages += f"\n  Priority : {state['escalation_priority']}"
            if state.get("escalation_reason"):
                messages += f"\n  Reason   : {state['escalation_reason']}"
        else:
            messages += "\nThis email does NOT warrant an escalation."
            if state.get("escalation_reason"):
                messages += f"\n  Reason   : {state['escalation_reason']}"

        state["messages"] = [messages]  # type: ignore[list-item]
        if timestamp:
            Path("output").mkdir(exist_ok=True)
            with open(f"output/email_extract_{timestamp}.md", "w+") as f:
                f.write(messages)
        else:
            logging.warning(f"{self.ParseEmailSummarizer.__name__} No timestampe information!")
        logging.debug(f"state: {state}")
        return state

    async def CreateGraph(self) -> CompiledStateGraph:
        """
        Compile application and test
        https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/
        https://langchain-ai.github.io/langgraph/concepts/low_level/#node-caching
        """
        logging.info(f"\n=== {self.CreateGraph.__name__} ===")
        try:
            cache_policy = CachePolicy(ttl=600) # 10 minutes
            await self._db_pool.open()
            if self._store is None:
                self._store = AsyncPostgresStore(self._db_pool, index={
                            "embed": OllamaEmbeddings(model=appconfig.EMBEDDING_MODEL, base_url=appconfig.OLLAMA_LOCAL_URI, num_ctx=appconfig.OLLAMA_CONTEXT_LENGTH, num_gpu=1, temperature=0),
                            "dims": appconfig.EMBEDDING_DIMENSIONS, # Note: Every time when this value changes, remove the store<foo> tables in the DB so that store.setup() runs to recreate them with the right dimensions.
                        }
                )
                await PostgreSQLStoreSetup(self._db_pool, self._store) # store is needed when creating the ReAct agent / StateGraph for InjectedStore to work
            if self._checkpointer is None:
                self._checkpointer = AsyncPostgresSaver(self._db_pool)
                await PostgreSQLCheckpointerSetup(self._db_pool, self._checkpointer)
            # This should be a custom subagent.
            graph_builder = StateGraph(EmailRAGState, ContextSchema)
            graph_builder.add_node("ParseEmail", self.ParseEmail, cache_policy = cache_policy)
            graph_builder.add_node("NeedsEscalation", self.NeedsEscalation, cache_policy = cache_policy)
            graph_builder.add_node("ParseEmailSummarizer", self.ParseEmailSummarizer, cache_policy = cache_policy)
            graph_builder.add_edge(START, "ParseEmail")
            graph_builder.add_edge("ParseEmail", "NeedsEscalation")
            graph_builder.add_edge("NeedsEscalation", "ParseEmailSummarizer")
            graph_builder.add_edge("ParseEmailSummarizer", END)
            self._parser_graph = graph_builder.compile(name=self._graphName, cache=InMemoryCache(), store = self._store, checkpointer = self._checkpointer)
            # Use it as a custom subagent
            self._parser_subagent = CompiledSubAgent(
                name = "Email Parser SubAgent",
                description = """Extract structured fields from a regulatory email.
                            This should be used when the email message comes from
                            a regulatory body or auditor regarding a property or
                            construction site that the company works on.

                            escalation_criteria is a description of which kinds of
                            notices require immediate escalation.
                        """,
                system_prompt = EMAIL_PARSER_INSTRUCTIONS,
                runnable = self._parser_graph
            )
            self._subagents = [self._parser_subagent]
            self._agent = create_deep_agent(
                model = self._llm,
                tools = [RAGMemoryManager, RAGMemorySearcher],
                backend = composite_backend,
                store = self._store,
                checkpointer = self._checkpointer,
                system_prompt = EMAIL_PROCESSING_INSTRUCTIONS,
                subagents = self._subagents
            )
            # self.ShowGraph()
        except ResourceExhausted as e:
            logging.exception(f"google.api_core.exceptions.ResourceExhausted")
        return self._agent
    
    def ShowGraph(self):
        show_graph(self._parser_graph, self._graphName) # This blocks
        show_graph(self._agent, self._agentName) # This blocks

    @atheris.instrument_func
    async def Chat(self, criteria, email_state) -> List[str]:
        logging.info(f"\n=== {self.Chat.__name__} ===")
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        message = f"[Timestamp: {timestamp}]\nThe escalation criteria is: {criteria}\n"
        if "escalation_dollar_criteria" in email_state:
            message += f"The escalation dollar criteria is: {email_state['escalation_dollar_criteria']}\n"
        if "escalation_emails" in email_state:
            message += f"Escalation emails: {', '.join(email_state['escalation_emails'])}\n"
        message += f"Here's the email: {email_state['email']}"
        result: List[str] = []
        config = RunnableConfig(run_name="Email RAG", configurable={"thread_id": uuid7str(), "user_id": uuid7str(),  "timestamp": timestamp})
        async for chunk in self._agent.astream(
            {"messages": [{"role": "user", "content": message}]},
            stream_mode="values",
            config = config
        ):
            # Print thinking if available
            last = chunk["messages"][-1]
            if isinstance(last, dict) and "thinking" in last:
                if not self._in_thinking:
                    self._in_thinking = True
                    logging.debug("Thinking:\n")
                logging.debug(last["thinking"])
            else:
                if self._in_thinking:
                    logging.debug("\n\nAnswer:\n")
                    self._in_thinking = False
                result.append(last)
                if hasattr(last, "pretty_print"):
                    last.pretty_print()
        return result[-1]

async def make_graph(config: RunnableConfig) -> CompiledStateGraph:
    return await EmailRAG(config).CreateGraph()

async def main(criteria):
    # httpx library is a dependency of LangGraph and is used under the hood to communicate with the AI models.
    """
    graph = checkpoint_graph.get_graph().draw_mermaid_png()
    # Save the PNG data to a file
    with open("/tmp/checkpoint_graph.png", "wb") as f:
        f.write(graph)
    img = Image.open("/tmp/checkpoint_graph.png")
    img.show()
    """
    rag = EmailRAG()
    await rag.CreateGraph()
    email_state = {
        "email": EMAILS[3],
        "escalation_dollar_criteria": 100_000,
        "escalation_emails": ["brog@abc.com", "bigceo@company.com"],
    }
    result = await rag.Chat(criteria, email_state)
    assert result
    ai_message = ChatMessage.from_langchain(result)
    assert not ai_message.tool_calls
    assert ai_message.content

@atheris.instrument_func
def FuzzEntryPoint(data):
    # Initialize the provider with raw bytes from the fuzzer
    fdp = atheris.FuzzedDataProvider(data)    
    # Consume structured data
    number = fdp.ConsumeIntInRange(100, 100000)
    criteria = fdp.ConsumeUnicodeNoSurrogates(128)
    asyncio.run(main(criteria))

if __name__ == "__main__":
    print(f"argv 1: {sys.argv}")
    if len(sys.argv) > 1 and any("-atheris_runs" in s for s in sys.argv):
        atheris.Setup(sys.argv, FuzzEntryPoint)
        atheris.instrument_all()    
        atheris.Fuzz()    
    else:
        asyncio.run(main("There's an immediate risk of electrical, water, or fire damage"))
