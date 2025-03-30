import ast, os, mlflow, mlflow.deployments, pandas
from pandas import DataFrame, Series
from langchain.chains import RetrievalQA
#from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.llms import Databricks
from mlflow.deployments import set_deployments_target
from mlflow.metrics.genai.metric_definitions import relevance
from src.rag_agent.VectorStore import VectorStore
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
def evaluate_chunk_size(chunk_size):
    vector_store = VectorStore(model="text-embedding-005", chunk_size=chunk_size, chunk_overlap=100)
    vector_store.load(_urls)
    #embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    #retriever = Chroma.from_documents(docs, embedding_function).as_retriever()
    mlflow.set_experiment(experiment_name = evaluate_chunk_size.__name__)
    mlflow.langchain.autolog()
    def retrieve_doc_ids(question: str) -> list[str]:
        docs = vector_store.retriever.get_relevant_documents(question)
        return [doc.metadata["source"] for doc in docs]

    def retriever_model_function(question_df: DataFrame) -> Series:
        return question_df["question"].apply(retrieve_doc_ids)

    with mlflow.start_run():
        return mlflow.evaluate(
          model=retriever_model_function,
          data=_eval_data,
          model_type="retriever",
          targets="source",
          evaluators="default",
        )

if __name__ == "__main__":
    result = evaluate_chunk_size(1000)
    print(f"1000 chunk_size: {result.tables['eval_results_table']}")
    result = evaluate_chunk_size(2000)
    print(f"2000 chunk_size: {result.tables['eval_results_table']}")