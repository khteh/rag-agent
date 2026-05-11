import mlflow, os, pandas as pd, json
from asyncio import Queue, run, create_task
from psycopg_pool import AsyncConnectionPool, ConnectionPool
from urllib import parse
from mlflow.genai.scorers import Correctness, Guidelines
#from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from src.Infrastructure.VectorStore import VectorStore
from src.config import config as appconfig
eval_data = [
    {
        "query": "What is MLOps?",
        "expected_docs": ["https://www.databricks.com/blog/mlops-frameworks-complete-guide-tools-and-platforms-production-ml"],
    },
    {
        "query":  "What is MLflow?",
        "expected_docs": ["https://mlflow.org/docs/latest/index.html"],
    },
    {
        "query": "What is the Task Decomposition?",
        "expected_docs": ["https://lilianweng.github.io/posts/2023-06-23-agent/"],
    }
    # ... 30-100 queries for production benchmarks
]
eval_df = pd.DataFrame(eval_data)

def benchmark_retriever(
    run_name: str,
    retriever_fn,
    eval_df: pd.DataFrame,
    extra_params: dict = None,
):
    """Benchmark any retriever function and log results to MLflow."""
    print("=== benchmark_retriever ===")
    # Wrap the retriever so mlflow.evaluate() can call it on each row
    def retriever_for_mlflow(df: pd.DataFrame) -> pd.Series:
        return df["query"].apply(retriever_fn)
    
    mlflow.langchain.autolog()
    with mlflow.start_run(run_name=run_name):
        if extra_params:
            mlflow.log_params(extra_params)

        # Run the retriever against every query and score the results
        results = mlflow.evaluate(
            model=retriever_for_mlflow,
            data=eval_df[["query", "expected_docs"]],
            model_type="retriever",
            targets="expected_docs",
            evaluators="default",
            extra_metrics=[
                mlflow.metrics.precision_at_k(1),
                mlflow.metrics.precision_at_k(3),
                mlflow.metrics.precision_at_k(5),
                mlflow.metrics.recall_at_k(1),
                mlflow.metrics.recall_at_k(3),
                mlflow.metrics.recall_at_k(5),
                mlflow.metrics.ndcg_at_k(3),
                mlflow.metrics.ndcg_at_k(5),
            ],
        )
        return results

def evaluate_embedding(vector_store: VectorStore):
    print("=== evaluate_embedding ===")
    # All benchmark runs will be grouped under this experiment
    mlflow.set_experiment("vector-search-benchmark")

    benchmark_retriever(
        run_name="evaluate_embedding",
        retriever_fn=lambda q: vector_store.asimilarity_search(q, top_k=5),  # index uses gte-large
        eval_df=eval_df,
        extra_params={"vector_store": "postgresql vector", "embedding_model": appconfig.LLM_RAG_MODEL},
    )

def evaluate_k_nearest_neighbours(data):
    mlflow.set_experiment(evaluate_k_nearest_neighbours.__name__)
    mlflow.langchain.autolog()
    with mlflow.start_run() as run:
        return mlflow.models.evaluate(
            data=data,
            targets="source",
            predictions="outputs",
            evaluators="default",
            extra_metrics=[
                mlflow.metrics.precision_at_k(1),
                mlflow.metrics.precision_at_k(2),
                mlflow.metrics.precision_at_k(3),
                mlflow.metrics.recall_at_k(1),
                mlflow.metrics.recall_at_k(2),
                mlflow.metrics.recall_at_k(3),
                mlflow.metrics.ndcg_at_k(1),
                mlflow.metrics.ndcg_at_k(2),
                mlflow.metrics.ndcg_at_k(3),
            ],
        )

async def main():
    db_pool = AsyncConnectionPool(
        conninfo = appconfig.POSTGRESQL_DATABASE_URI,
        max_size = appconfig.DB_MAX_CONNECTIONS,
        kwargs = appconfig.connection_kwargs,
        open = True
    )
    vector_store = VectorStore(db_pool)
    await vector_store.CreateResources()
    result = evaluate_embedding(vector_store)
    # To validate the results of a different model, comment out the above line and uncomment the below line:
    # result2 = evaluate_embedding(SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))

    eval_results_of_retriever_df_bge = result.tables["eval_results_table"]
    # To validate the results of a different model, comment out the above line and uncomment the below line:
    # eval_results_of_retriever_df_MiniLM = result2.tables["eval_results_table"]
    print(f"Embeddings Evaluation result: {eval_results_of_retriever_df_bge}")
    result = evaluate_k_nearest_neighbours(eval_results_of_retriever_df_bge)
    print(f"evaluate_k_nearest_neighbours result: {result.tables['eval_results_table']}")

if __name__ == "__main__":
    #with open('/etc/ragagent_config.json', 'r') as f:
    #    config = json.load(f)
    #mlflow.set_tracking_uri(f"postgresql+psycopg://{os.environ.get('DB_USERNAME')}:{parse.quote(os.environ.get('DB_PASSWORD'))}@{config['DB_HOST']}/MLFlow")
    print(f"MLFlow tracking URI: {appconfig.MLFLOW_URI}")
    mlflow.set_tracking_uri(appconfig.MLFLOW_URI)
    run(main())