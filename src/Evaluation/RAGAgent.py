import mlflow, pandas
from asyncio import Queue, run, create_task
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
#from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from mlflow.metrics.genai import relevance
from psycopg_pool import AsyncConnectionPool, ConnectionPool
from src.Infrastructure.VectorStore import VectorStore
from src.config import config
# https://mlflow.org/docs/latest/api_reference/python_api/mlflow.metrics.html
_eval_data = pandas.DataFrame(
  {
      "question": [
          "What is MLOps?",
          "What is MLflow?",
          "What is Databricks?",
          "How to serve a model on Databricks?",
          "How to enable MLflow Autologging for my workspace by default?",
          "What is the Task Decomposition?",
          "What is the standard method for Task Decomposition?",
          "What are the common extensions of standard method for Task Decomposition?"
      ],
  }
)
vectorStore: VectorStore = None

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
          _eval_data,
          predictions="result",
          extra_metrics=[relevance_metric, mlflow.metrics.latency()],
          evaluator_config={
              "col_mapping": {
                  "inputs": "questions",
                  "context": "source_documents",
              }
          },
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
    mlflow.set_tracking_uri(config.MLFLOW_URI)
    #vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    run(main())