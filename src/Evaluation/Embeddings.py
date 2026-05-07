import mlflow, os, pandas, json
from asyncio import Queue, run, create_task
from psycopg_pool import AsyncConnectionPool, ConnectionPool
from urllib import parse
from mlflow.genai.scorers import Correctness, Guidelines
#from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from src.Infrastructure.VectorStore import VectorStore
from src.config import config as appconfig
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
      "source": [
          ["https://www.databricks.com/blog/mlops-frameworks-complete-guide-tools-and-platforms-production-ml"],
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
# Define a function that runs predictions.
def predict_fn(question: str) -> str:
  response = openai.OpenAI().chat.completions.create(
      model="gpt-5-mini",
      messages=[
          {"role": "system", "content": "Answer the following question in two sentences"},
          {"role": "user", "content": question},
      ],
  )
  return response.choices[0].message.content

def evaluate_embedding(vector_store: VectorStore):
    # For VertexAI, use VertexAIEmbeddings, model="text-embedding-005"; "gemini-2.0-flash" model_provider="google_genai"
    def retrieve_doc_ids(question: str) -> list[str]:
        docs = vector_store.retriever.invoke(question)
        return [doc.metadata["source"] for doc in docs]
    def retriever_model_function(question_df: pandas.DataFrame) -> pandas.Series:
        return question_df["question"].apply(retrieve_doc_ids)
    mlflow.set_experiment(evaluate_embedding.__name__)
    mlflow.langchain.autolog()
  
    #with mlflow.start_run():
    #    return mlflow.models.evaluate(
    #      model=retriever_model_function,
    #      data=_eval_data,
    #      model_type="retriever",
    #      targets="source",
    #      evaluators="default",
    #    )
    mlflow.genai.evaluate(data=_eval_data, predict_fn=predict_fn, scorers=[
        # Built-in LLM judge
        Correctness()])

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
    with open('/etc/ragagent_config.json', 'r') as f:
        config = json.load(f)
    mlflow.set_tracking_uri(f"postgresql+psycopg://{os.environ.get('DB_USERNAME')}:{parse.quote(os.environ.get('DB_PASSWORD'))}@{config['DB_HOST']}/MLFlow")
    run(main())