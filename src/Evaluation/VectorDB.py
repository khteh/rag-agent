import ast, os, mlflow, mlflow.deployments, pandas
from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.databricks import DatabricksEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms import Databricks
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import InMemoryVectorStore
from mlflow.deployments import set_deployments_target
from mlflow.metrics.genai.metric_definitions import relevance

EVALUATION_DATASET_PATH = "https://raw.githubusercontent.com/mlflow/mlflow/master/examples/llms/RAG/static_evaluation_dataset.csv"

synthetic_eval_data = pandas.read_csv(EVALUATION_DATASET_PATH)

# Load the static evaluation dataset from disk and deserialize the source and retrieved doc ids
synthetic_eval_data["source"] = synthetic_eval_data["source"].apply(ast.literal_eval)
synthetic_eval_data["retrieved_doc_ids"] = synthetic_eval_data["retrieved_doc_ids"].apply(
  ast.literal_eval
)