import mlflow, pandas
from asyncio import Queue, run, create_task
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
#from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from mlflow.entities import Document
from mlflow.genai.scorers import (
    Correctness, Guidelines,
    RelevanceToQuery,
    RetrievalGroundedness,
    RetrievalSufficiency,
)
from mlflow.metrics.genai import relevance
from psycopg_pool import AsyncConnectionPool, ConnectionPool
from src.Infrastructure.VectorStore import VectorStore
from src.config import config
# https://mlflow.org/docs/latest/api_reference/python_api/mlflow.metrics.html
eval_data = [
    {
        "inputs": {"question": "What is MLOps?"},
        "expectations": {
            "expected_facts": [
                "machine learning operations",
                "applies automation, version control, and continuous delivery — to the full machine learning lifecycle",
            ]
        },
    },
    {
        "inputs": {"question": "What is MLflow?"},
        "expectations": {
            "expected_facts": [
                "open-source platform",
                "managing the ML lifecycle",
            ]
        },
    },
    {
        "inputs": {"question": "What is the Task Decomposition?"},
        "expectations": {
            "expected_facts": [
                "breaks down large tasks into smaller, manageable subgoals, enabling efficient handling of complex tasks.",
                "Chain of thought",
                "Tree of thought",
            ]
        },
    }
]
eval_df = pandas.DataFrame(eval_data)
vectorStore: VectorStore = None

@mlflow.trace(span_type="RETRIEVER")
async def retrieve(question: str, n_results: int = 3) -> list[Document]:
    results = vectorStore.invoke(query_texts=[question], n_results=n_results)
    return [
        Document(page_content=doc, metadata={"doc_id": doc_id})
        for doc, doc_id in zip(results["documents"][0], results["ids"][0])
    ]

@mlflow.trace
async def rag_answer(question: str) -> str:
    docs = retrieve(question)
    context = "\n".join(doc.page_content for doc in docs)

    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[
            {
                "role": "system",
                "content": f"Answer the question based only on the following context:\n\n{context}",
            },
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content

def predict_fn(question):
    return rag_answer(question)

def model(input_df):
    # For VertexAI, use VertexAIEmbeddings, model="text-embedding-005"; "gemini-2.0-flash" model_provider="google_genai"
    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentence maximum and keep the answer concise. "
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )    
    question_answer_chain = create_stuff_documents_chain(model, prompt)
    chain = create_retrieval_chain(vectorStore.vector_store.as_retriever(fetch_k=3), question_answer_chain)
    return input_df["questions"].map(chain).tolist()

def evaluate_rag():
    relevance_metric = relevance()
    print(f"Relevance details: {relevance_metric.metric_details}")
    mlflow.set_experiment(evaluate_rag.__name__)
    mlflow.langchain.autolog()
    if config.OLLAMA_CLOUD_URI:
        llm = init_chat_model(config.LLM_RAG_MODEL, model_provider=config.MODEL_PROVIDER, base_url=config.OLLAMA_CLOUD_URI, api_key=config.OLLAMA_API_KEY, streaming=True, temperature=0, think="high")
    else:
        llm = init_chat_model(config.LLM_RAG_MODEL, model_provider=config.MODEL_PROVIDER, api_key=config.OLLAMA_API_KEY, streaming=True, temperature=0, think="high")
  
    with mlflow.start_run():
      results = mlflow.genai.evaluate(
            llm,
            data=eval_data,
            predict_fn=predict_fn,
            scorers=[
                RelevanceToQuery(),
                RetrievalGroundedness(),
                RetrievalSufficiency(),
            ],
      )
    print(results.metrics)
    print(results.tables["eval_results_table"])

async def main():
    db_pool = AsyncConnectionPool(
        conninfo = config.POSTGRESQL_DATABASE_URI,
        max_size = config.DB_MAX_CONNECTIONS,
        kwargs = config.connection_kwargs,
        open = True
    )
    vector_store = VectorStore(db_pool)
    await vector_store.CreateResources()
    evaluate_rag(vector_store)

if __name__ == "__main__":
    #mlflow.set_tracking_uri("http://localhost:8080") $ pr mlflow server --host localhost --port 8080
    print(f"MLFlow tracking URI: {config.MLFLOW_URI}")
    mlflow.set_tracking_uri(config.MLFLOW_URI)
    #vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    run(main())