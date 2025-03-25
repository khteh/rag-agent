import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from .prompts import cypher_generation_template, qa_generation_template
load_dotenv()

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

graph.refresh_schema()

cypher_generation_prompt = PromptTemplate(
    input_variables=["schema", "question"], template=cypher_generation_template
)

qa_generation_prompt = PromptTemplate(
    input_variables=["context", "question"], template=qa_generation_template
)
"""
        self._llm = init_chat_model("gemini-2.0-flash", model_provider="google_vertexai", streaming=True).bind_tools(TOOLS)
        # https://python.langchain.com/docs/integrations/chat/google_vertex_ai_palm/
        self._llm = ChatVertexAI(
                        model="gemini-2.0-flash",
                        temperature=0,
                        max_tokens=None,
                        max_retries=6,
                        stop=None,
                        streaming=True
                    )
"""
hospital_cypher_chain = GraphCypherQAChain.from_llm(
    cypher_llm = init_chat_model("gemini-2.0-flash", model_provider="google_vertexai", streaming=True, temperature=0),
    qa_llm = init_chat_model("gemini-2.0-flash", model_provider="google_vertexai", streaming=True, temperature=0),
    graph = graph,
    verbose = True, # Whether intermediate steps your chain performs should be printed.
    qa_prompt = qa_generation_prompt,
    cypher_prompt = cypher_generation_prompt,
    validate_cypher = True,
    top_k = 100,
)