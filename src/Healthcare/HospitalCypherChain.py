import os, logging
from langchain.chat_models import init_chat_model
from langchain_neo4j import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from .prompts import cypher_generation_template, qa_generation_template
from src.config import config

graph = Neo4jGraph(
    url=config.NEO4J_URI,
    username=config.NEO4J_USERNAME,
    password=config.NEO4J_PASSWORD,
)
graph.refresh_schema()
cypher_generation_prompt = PromptTemplate(
    input_variables=["schema", "question"], template=cypher_generation_template
)
qa_generation_prompt = PromptTemplate(
    input_variables=["context", "question"], template=qa_generation_template
)
hospital_cypher_chain = GraphCypherQAChain.from_llm(
    cypher_llm = init_chat_model("llama3.3", model_provider="ollama", base_url=config.OLLAMA_URI, streaming=True, temperature=0),
    qa_llm = init_chat_model("llama3.3", model_provider="ollama", base_url=config.OLLAMA_URI, streaming=True, temperature=0),
    graph = graph,
    verbose = True, # Whether intermediate steps your chain performs should be printed.
    qa_prompt = qa_generation_prompt,
    cypher_prompt = cypher_generation_prompt,
    validate_cypher = True,
    top_k = 100,
)