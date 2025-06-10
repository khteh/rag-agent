from langchain_core.tools import InjectedToolArg, tool
from typing_extensions import Annotated
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate, SystemMessagePromptTemplate
from langchain_community.vectorstores import Neo4jVector
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
from src.common.configuration import Configuration
from src.config import config as appconfig
from src.utils.ModelString import split_model_and_provider
# For VertexAI, use VertexAIEmbeddings, model="text-embedding-005"; "gemini-2.0-flash" model_provider="google_genai"
@tool(description ="""Useful when you need to answer questions
        about patient experiences, feelings, or any other qualitative
        question that could be answered about a patient using semantic
        search. Not useful for answering objective questions that involve
        counting, percentages, aggregations, or listing facts. Use the
        entire prompt as input to the tool. For instance, if the prompt is
        "Are patients satisfied with their care?", the input should be
        "Are patients satisfied with their care?".
        """)
async def HealthcareReview(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> dict[str, float]:
    neo4j_vector_index = Neo4jVector.from_existing_graph(
        embedding = OllamaEmbeddings(model=appconfig.EMBEDDING_MODEL, base_url=appconfig.OLLAMA_URI, num_ctx=8192, num_gpu=1, temperature=0),
        url = appconfig.NEO4J_URI,
        username = appconfig.NEO4J_USERNAME,
        password = appconfig.NEO4J_PASSWORD,
        index_name = "reviews",
        node_label = "Review",
        text_node_properties=[
            "physician_name",
            "patient_name",
            "text",
            "hospital_name",
        ],
        embedding_node_property="embedding",
    )
    configuration = Configuration.from_runnable_config(config)
    review_system_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(input_variables=["context"], template=configuration.review_prompt)
    )
    review_human_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(input_variables=["question"], template="{question}")
    )
    messages = [review_system_prompt, review_human_prompt]
    review_prompt = ChatPromptTemplate(
        input_variables=["context", "question"], messages=messages
    )
    reviews_vector_chain = RetrievalQA.from_chain_type(
        llm = init_chat_model(appconfig.LLM_RAG_MODEL, model_provider="ollama", base_url=appconfig.OLLAMA_URI, streaming=True, temperature=0),
        chain_type = "stuff", # pass all k reviews to the prompt.
        retriever = neo4j_vector_index.as_retriever(k=3),
    )
    reviews_vector_chain.combine_documents_chain.llm_chain.prompt = review_prompt    
    return await reviews_vector_chain.ainvoke(query)

@tool(description="""Useful for answering questions about patients,
        physicians, hospitals, insurance payers, patient review
        statistics, and hospital visit details. Use the entire prompt as
        input to the tool. For instance, if the prompt is "How many visits
        have there been?", the input should be "How many visits have
        there been?""")
async def HealthcareCypher(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> dict[str, float]:
    with Neo4jGraph(
        url = appconfig.NEO4J_URI,
        username = appconfig.NEO4J_USERNAME,
        password = appconfig.NEO4J_PASSWORD,
    ) as graph:
        configuration = Configuration.from_runnable_config(config)
        cypher_generation_prompt = PromptTemplate(
            input_variables=["schema", "question"], template=configuration.cypher_generation_prompt
        )
        qa_generation_prompt = PromptTemplate(
            input_variables=["context", "question"], template=configuration.qa_generation_prompt
        )
        hospital_cypher_chain = GraphCypherQAChain.from_llm(
            cypher_llm = init_chat_model(appconfig.LLM_RAG_MODEL, model_provider="ollama", base_url=appconfig.OLLAMA_URI, streaming=True, temperature=0),
            qa_llm = init_chat_model(appconfig.LLM_RAG_MODEL, model_provider="ollama", base_url=appconfig.OLLAMA_URI, streaming=True, temperature=0),
            graph = graph,
            verbose = appconfig.LOGLEVEL == "DEBUG", # Whether intermediate steps your chain performs should be printed.
            qa_prompt = qa_generation_prompt,
            cypher_prompt = cypher_generation_prompt,
            validate_cypher = True,
            top_k = 100,
            allow_dangerous_requests = True # https://python.langchain.com/docs/security/
        )    
        return await hospital_cypher_chain.ainvoke(query)