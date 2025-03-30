import ast, os, mlflow, mlflow.deployments, pandas
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import DatabricksEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Databricks
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from mlflow.deployments import set_deployments_target
from mlflow.metrics.genai.metric_definitions import relevance

EVALUATION_DATASET_PATH = "https://raw.githubusercontent.com/mlflow/mlflow/master/examples/llms/RAG/static_evaluation_dataset.csv"

synthetic_eval_data = pandas.read_csv(EVALUATION_DATASET_PATH) # question,source,retrieved_doc_ids

# Load the static evaluation dataset from disk and deserialize the source and retrieved doc ids
synthetic_eval_data["source"] = synthetic_eval_data["source"].apply(ast.literal_eval)
synthetic_eval_data["retrieved_doc_ids"] = synthetic_eval_data["retrieved_doc_ids"].apply(
  ast.literal_eval
)

print("\nsynthetic_eval_data['source']:")
print(f"id: {id(synthetic_eval_data["source"])}, ndim: {synthetic_eval_data['source'].ndim}, size: {synthetic_eval_data['source'].size}, shape: {synthetic_eval_data['source'].shape}")
print(synthetic_eval_data["source"].describe())
synthetic_eval_data["source"].info()
print(synthetic_eval_data["source"].head())

print("\nsynthetic_eval_data['retrieved_doc_ids']:")
print(f"id: {id(synthetic_eval_data["retrieved_doc_ids"])}, ndim: {synthetic_eval_data['retrieved_doc_ids'].ndim}, size: {synthetic_eval_data['retrieved_doc_ids'].size}, shape: {synthetic_eval_data['retrieved_doc_ids'].shape}")
print(synthetic_eval_data["retrieved_doc_ids"].describe())
synthetic_eval_data["retrieved_doc_ids"].info()
print(synthetic_eval_data["retrieved_doc_ids"].head())
