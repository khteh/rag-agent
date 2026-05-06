import mlflow, asyncio
from asyncio import Queue, run, create_task
from psycopg_pool import AsyncConnectionPool, ConnectionPool
from pandas import DataFrame, Series
#from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from src.Infrastructure.VectorStore import VectorStore
from src.config import config
_urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    "https://mlflow.org/docs/latest/index.html",
    "https://mlflow.org/docs/latest/tracking/autolog.html",
    "https://mlflow.org/docs/latest/getting-started/tracking-server-overview/index.html",
    "https://mlflow.org/docs/latest/python_api/mlflow.deployments.html",        
]

_eval_data = DataFrame(
  {
      "question": [
          "What is MLflow?",
          "What is Databricks?",
          "How to serve a model on Databricks?",
          "How to enable MLflow Autologging for my workspace by default?",
          "What is the Task Decomposition?",
          "What is the standard method for Task Decomposition?",
          "What are the common extensions of standard method for Task Decomposition?"
      ],
      "source": [
          ["https://mlflow.org/docs/latest/index.html"],
          ["https://mlflow.org/docs/latest/getting-started/tracking-server-overview/index.html"],
          ["https://mlflow.org/docs/latest/python_api/mlflow.deployments.html"],
          ["https://mlflow.org/docs/latest/tracking/autolog.html"],
          ["https://lilianweng.github.io/posts/2023-06-23-agent/"],
          ["https://lilianweng.github.io/posts/2023-06-23-agent/"],
          ["https://lilianweng.github.io/posts/2023-06-23-agent/"]
      ],
  }
)
async def evaluate_chunk_size(vector_store: VectorStore, chunk_size):
    # For VertexAI, use VertexAIEmbeddings, model="text-embedding-005"; "gemini-2.0-flash" model_provider="google_genai"
    vector_store.LoadDocuments(_urls, chunk_size, 0)
    mlflow.set_experiment(evaluate_chunk_size.__name__)
    mlflow.langchain.autolog()
    def retrieve_doc_ids(question: str) -> list[str]:
        docs = vector_store.retriever.get_relevant_documents(question)
        return [doc.metadata["source"] for doc in docs]

    def retriever_model_function(question_df: DataFrame) -> Series:
        return question_df["question"].apply(retrieve_doc_ids)

    with mlflow.start_run():
        return mlflow.models.evaluate(
          model=retriever_model_function,
          data=_eval_data,
          model_type="retriever",
          targets="source",
          evaluators="default",
        )

async def main():
    db_pool = AsyncConnectionPool(
        conninfo = config.POSTGRESQL_DATABASE_URI,
        max_size = config.DB_MAX_CONNECTIONS,
        kwargs = config.connection_kwargs,
        open = True
    )
    vector_store = VectorStore(db_pool)
    await vector_store.CreateResources()
    result = evaluate_chunk_size(vector_store, 1000)
    print(f"1000 chunk_size: {result.tables['eval_results_table']}")
    result = evaluate_chunk_size(vector_store, 2000)
    print(f"2000 chunk_size: {result.tables['eval_results_table']}")

if __name__ == "__main__":
    run(main())