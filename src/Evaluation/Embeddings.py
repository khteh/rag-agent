import mlflow, logging, pandas as pd, json
from asyncio import Queue, run, create_task
from psycopg_pool import AsyncConnectionPool, ConnectionPool
from urllib import parse
from mlflow.entities import Document
from mlflow.genai.scorers import scorer
from mlflow.genai.scorers import (
    Correctness, Guidelines,
    RelevanceToQuery,
    RetrievalGroundedness,
    RetrievalSufficiency,
)
#from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from src.Infrastructure.VectorStore import VectorStore
from src.config import config as appconfig
eval_data = [
    {
        "inputs": {"question": "What is MLOps?"},
        "expected_docs": ["https://www.databricks.com/blog/mlops-frameworks-complete-guide-tools-and-platforms-production-ml"],
    },
    {
        "inputs": {"question": "What is MLFlow?"},
        "expected_docs": ["https://mlflow.org/docs/latest/index.html"],
    },
    {
        "inputs": {"question": "What is the Task Decomposition?"},
        "expected_docs": ["https://lilianweng.github.io/posts/2023-06-23-agent/"],
    }
    # ... 30-100 queries for production benchmarks
]
eval_df = pd.DataFrame(eval_data)
vectorStore: VectorStore = None

@scorer
def exact_match(outputs: dict, expectations: dict) -> bool:
    return outputs['source'] == expectations["expected_docs"]

@mlflow.trace(span_type="RETRIEVER")
async def predict_fn(question: str, n_results: int = 3) -> list[Document]:
    logging.debug(f"\n=== {predict_fn.__name__} ===")
    global vectorStore
    results = await vectorStore.retriever.ainvoke(question, n_results=n_results)
    """
    2026-05-11 12:48:40 DEBUG    results: [Document(id='069fb2d8-17e4-7f21-8000-3b97e9a59a21', metadata={'source': 'https://www.databricks.com/blog/mlops-frameworks-complete-guide-tools-and-platforms-production-ml', 'start_index': 0}, 
    page_content='Getting a machine learning model to perform well in a notebook is only half the battle. Moving that model into a reliable, scalable production environment — and keeping it performing over time — is where most teams struggle. That gap between experimentation and reliable deployment is exactly what MLOps frameworks are designed to close.MLOps (machine learning operations) has emerged as a discipline that applies MLOps principles — automation, version control, and continuous delivery — to the full machine learning lifecycle. The right framework can mean the difference between models that stagnate in development and models that drive real business value at scale. Yet with dozens of options available, from lightweight open-source tools to full-featured enterprise MLOps platforms, choosing the right fit requires a clear understanding of what each layer of the stack actually does'), Document(id='069fb2d8-17e5-71a7-8000-f61f3ccbbff1', metadata={'source': 'https://www.databricks.com/blog/mlops-frameworks-complete-guide-tools-and-platforms-production-ml', 'start_index': 885}, page_content=".This guide breaks down the most widely adopted MLOps frameworks, the core components they address, and how to evaluate them against your team's specific needs. Whether you're a startup building your first production pipeline or a large enterprise managing hundreds of ML models across multiple clouds, there's a framework architecture designed for your situation.Why MLOps Frameworks Exist — and What They Actually SolveThe challenge of machine learning operations goes deeper than simple DevOps automation. ML workflows involve dynamic datasets, non-deterministic training runs, complex model versioning requirements, and the ongoing need for model monitoring after deployment. Traditional software engineering practices, while necessary, are not sufficient on their own.Consider a typical machine learning project without structured tooling. Data scientists run dozens of experiments in isolation, logging parameters manually or not at all"), Document(id='069fb2d8-17e5-739c-8000-75b6dfa82b81', metadata={'source': 'https://www.databricks.com/blog/mlops-frameworks-complete-guide-tools-and-platforms-production-ml', 'start_index': 2636}, page_content=".Core Components of Any MLOps FrameworkBefore comparing specific tools, it's worth understanding what capabilities a complete MLOps workflow needs to support.Experiment tracking is the foundation. ML engineers and data scientists run hundreds of training iterations varying algorithms, hyperparameter tuning configurations, and feature engineering approaches. Without systematic tracking of metrics, parameters, and code versions linked to each run, reproducible results are impossible. Experiment tracking tools create a searchable audit trail of every training run, enabling teams to compare model performance across iterations and confidently promote the best version.Model versioning and the model registry extend version control beyond code to models themselves. A model registry acts as the central store where trained ML models are catalogued, versioned, and transitioned through lifecycle stages — from staging and validation through production and archival"), Document(id='069fb2d7-581f-794f-8000-68b6e334c7f3', metadata={'source': 'https://mlflow.org/docs/latest/ml/', 'start_index': 888}, page_content=".\nGetting Started with MLflow for ML Models\u200b\nThis page covers MLflow's tools for traditional machine learning and deep learning: ML experiment\ntracking, model versioning, model deployment, and model evaluation. If you're building agents and LLM applications, see MLflow for LLMs and Agents.\nIf this is your first time exploring MLflow for MLOps, the tutorials and guides here are a great place to start.\nQuickstartA quick guide to learn the basics of MLflow for MLOps by training a simple scikit-learn modelStart learning →MLflow for Agents & LLMsA walkthrough of MLflow's Agent and LLM capabilities, including tracing, evaluation, and prompt managementStart building →Deep Learning GuideA hands-on tutorial on how to use MLflow for ML to track deep learning model training with PyTorchStart training →\nMLflow for ML Models: Core Capabilities\u200b\nMLflow for ML Models provides comprehensive support for traditional machine learning and deep learning workflows")]
    """
    logging.debug(f"results: {results}")
    # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.entities.html#mlflow.entities.Document
    doc = Document(id=results[0].id, page_content=results[0].page_content, metadata={"source": results[0].metadata['source']})
    logging.debug(f"doc: {doc}")
    return doc

def evaluate_embedding():
    logging.debug(f"\n=== {evaluate_embedding.__name__} ===")
    # All benchmark runs will be grouped under this experiment
    mlflow.set_experiment(evaluate_embedding.__name__)
    mlflow.langchain.autolog()
    results = mlflow.genai.evaluate(
        data=eval_df,
        predict_fn=predict_fn,
        scorers=[
            exact_match,
            #RelevanceToQuery(),
            #RetrievalGroundedness(),
            #RetrievalSufficiency(),
        ]
    )
    return results

def evaluate_k_nearest_neighbours(data):
    logging.debug(f"\n=== {evaluate_k_nearest_neighbours.__name__} ===")
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
    global vectorStore
    vectorStore = VectorStore(db_pool)
    await vectorStore.CreateResources()
    result = evaluate_embedding()
    # To validate the results of a different model, comment out the above line and uncomment the below line:
    # result2 = evaluate_embedding(SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))

    eval_results_of_retriever_df_bge = result.tables["eval_results_table"]
    # To validate the results of a different model, comment out the above line and uncomment the below line:
    # eval_results_of_retriever_df_MiniLM = result2.tables["eval_results_table"]
    logging.debug(f"Embeddings Evaluation result: {eval_results_of_retriever_df_bge}")
    result = evaluate_k_nearest_neighbours(eval_results_of_retriever_df_bge)
    logging.debug(f"evaluate_k_nearest_neighbours result: {result.tables['eval_results_table']}")

if __name__ == "__main__":
    #with open('/etc/ragagent_config.json', 'r') as f:
    #    config = json.load(f)
    #mlflow.set_tracking_uri(f"postgresql+psycopg://{os.environ.get('DB_USERNAME')}:{parse.quote(os.environ.get('DB_PASSWORD'))}@{config['DB_HOST']}/MLFlow")
    logging.debug(f"MLFlow tracking URI: {appconfig.MLFLOW_URI}")
    mlflow.set_tracking_uri(appconfig.MLFLOW_URI)
    run(main())