import ast, os, mlflow, mlflow.deployments, pandas
from langchain.chains import RetrievalQA
#from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.llms import Databricks
from mlflow.deployments import set_deployments_target
from mlflow.metrics.genai.metric_definitions import relevance
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

_eval_data = pandas.DataFrame(
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

def evaluate_embedding():
    # For VertexAI, use VertexAIEmbeddings, model="text-embedding-005"; "gemini-2.0-flash" model_provider="google_genai"
    vector_store = VectorStore(model=config.EMBEDDING_MODEL, chunk_size=1000, chunk_overlap=0)
    vector_store.load(_urls)
    def retrieve_doc_ids(question: str) -> list[str]:
        docs = vector_store.retriever.get_relevant_documents(question)
        return [doc.metadata["source"] for doc in docs]
    def retriever_model_function(question_df: pandas.DataFrame) -> pandas.Series:
        return question_df["question"].apply(retrieve_doc_ids)
    mlflow.set_experiment(evaluate_embedding.__name__)
    mlflow.langchain.autolog()
  
    with mlflow.start_run():
        return mlflow.evaluate(
          model=retriever_model_function,
          data=_eval_data,
          model_type="retriever",
          targets="source",
          evaluators="default",
        )

def evaluate_k_nearest_neighbours(data):
    mlflow.set_experiment(evaluate_k_nearest_neighbours.__name__)
    mlflow.langchain.autolog()
    with mlflow.start_run() as run:
        return mlflow.evaluate(
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

if __name__ == "__main__":
    result = evaluate_embedding()
    # To validate the results of a different model, comment out the above line and uncomment the below line:
    # result2 = evaluate_embedding(SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))

    eval_results_of_retriever_df_bge = result.tables["eval_results_table"]
    # To validate the results of a different model, comment out the above line and uncomment the below line:
    # eval_results_of_retriever_df_MiniLM = result2.tables["eval_results_table"]
    print(f"Embeddings Evaluation result: {eval_results_of_retriever_df_bge}")
    result = evaluate_k_nearest_neighbours(eval_results_of_retriever_df_bge)
    print(f"evaluate_k_nearest_neighbours result: {result.tables['eval_results_table']}")

