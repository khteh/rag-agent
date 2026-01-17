import logging
from langchain_core.tools import InjectedToolArg, tool
from typing_extensions import Annotated
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate, SystemMessagePromptTemplate
from langchain_community.vectorstores import Neo4jVector
from langchain_ollama import OllamaEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from neo4j.exceptions import CypherSyntaxError
from src.common.Configuration import Configuration
from src.config import config as appconfig
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
    logging.info(f"\n=== HealthcareReview ===")
    try:
        neo4j_vector_index = Neo4jVector.from_existing_graph(
            embedding = OllamaEmbeddings(model=appconfig.EMBEDDING_MODEL, base_url=appconfig.BASE_URI, num_ctx=8192, num_gpu=1, temperature=0),
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
            prompt=PromptTemplate(input_variables=["input"], template="{input}")
        )
        messages = [review_system_prompt, review_human_prompt]
        review_prompt = ChatPromptTemplate(
            input_variables=["context", "input"], messages=messages
        )
        if appconfig.BASE_URI:
            llm = init_chat_model(appconfig.LLM_RAG_MODEL, model_provider=appconfig.MODEL_PROVIDER, base_url=appconfig.BASE_URI, streaming=True, temperature=0)
        else:
            llm = init_chat_model(appconfig.LLM_RAG_MODEL, model_provider=appconfig.MODEL_PROVIDER, streaming=True, temperature=0)
        #create_stuff_documents_chain():
        #    prompt: Prompt template. Must contain input variable `"context"` (override by
        #    setting document_variable), which will be used for passing in the formatted
        #    documents.
        question_answer_chain = create_stuff_documents_chain(llm, review_prompt)
        reviews_vector_chain = create_retrieval_chain(neo4j_vector_index.as_retriever(k=3), question_answer_chain)
        logging.debug(f"HealthcareReview query: {query}")
        return await reviews_vector_chain.ainvoke({"input": query})
    except Exception as e:
        logging.exception(f"HealthcareReview Exception! {str(e)}, repr: {repr(e)}")
        return repr(e)

@tool(description="""Useful for answering questions about patients,
        physicians, hospitals, insurance payers, patient review
        statistics, and hospital visit details. Use the entire prompt as
        input to the tool. For instance, if the prompt is "How many visits
        have there been?", the input should be "How many visits have
        there been?""")
async def HealthcareCypher(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> dict[str, float]:
    logging.info(f"\n=== HealthcareCypher ===")
    with Neo4jGraph(
        url = appconfig.NEO4J_URI,
        username = appconfig.NEO4J_USERNAME,
        password = appconfig.NEO4J_PASSWORD,
    ) as graph:
        try:
            configuration = Configuration.from_runnable_config(config)
            cypher_generation_prompt = PromptTemplate(
                input_variables=["schema", "question"], template=configuration.cypher_generation_prompt
            )
            qa_generation_prompt = PromptTemplate(
                input_variables=["context", "question"], template=configuration.qa_generation_prompt
            )
            if appconfig.BASE_URI:
                llm = init_chat_model(appconfig.LLM_RAG_MODEL, model_provider=appconfig.MODEL_PROVIDER, base_url=appconfig.BASE_URI, streaming=True, temperature=0)
            else:
                llm = init_chat_model(appconfig.LLM_RAG_MODEL, model_provider=appconfig.MODEL_PROVIDER, streaming=True, temperature=0)
            hospital_cypher_chain = GraphCypherQAChain.from_llm( # 'GraphCypherQAChain' object does not support the context manager protocol"
                cypher_llm = llm,
                qa_llm = llm,
                graph = graph,
                verbose = appconfig.LOGLEVEL == "DEBUG", # Whether intermediate steps your chain performs should be printed.
                qa_prompt = qa_generation_prompt,
                cypher_prompt = cypher_generation_prompt,
                validate_cypher = True,
                top_k = 100,
                allow_dangerous_requests = True # https://python.langchain.com/docs/security/
            )
            return await hospital_cypher_chain.ainvoke(query)
        except CypherSyntaxError as syntax_error:
            logging.exception(f"HealthcareCypher CypherSyntaxError: {syntax_error}, repr: {repr(syntax_error)}")
            return syntax_error.message
        except Exception as e:
            logging.exception(f"HealthcareCypher Exception! {str(e)}, repr: {repr(e)}")
            return repr(e)
