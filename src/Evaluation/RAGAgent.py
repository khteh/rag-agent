import ast, os, mlflow, mlflow.deployments, pandas, vertexai
from datetime import datetime
from langchain.chat_models import init_chat_model
from langchain.chains import RetrievalQA
#from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.llms import Databricks
from mlflow.deployments import set_deployments_target
from mlflow.metrics.genai.metric_definitions import relevance
from langchain_core.runnables import RunnableConfig
from src.Infrastructure.VectorStore import VectorStore
from src.rag_agent.RAGAgent import RAGAgent
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
  }
)
def model(input_df):
    # For VertexAI, use VertexAIEmbeddings, model="text-embedding-005"; "gemini-2.0-flash" model_provider="google_genai"
    model = init_chat_model(config.LLM_RAG_MODEL, model_provider="ollama", base_url=config.OLLAMA_URI, streaming=True)
    vectorStore = VectorStore(model=config.EMBEDDING_MODEL, chunk_size=1000, chunk_overlap=0)
    qa = RetrievalQA.from_chain_type(
          llm = model,
          chain_type="stuff",
          retriever=vectorStore.as_retriever(fetch_k=3),
          return_source_documents=True,
    )
    return input_df["questions"].map(qa).tolist()

def evaluate_rag():
    relevance_metric = relevance(
      model="endpoints:/databricks-llama-2-70b-chat"
    )  # You can also use any model you have hosted on Databricks, models from the Marketplace or models in the Foundation model API
    mlflow.set_experiment(evaluate_rag.__name__)
    mlflow.langchain.autolog()
  
    with mlflow.start_run():
      results = mlflow.evaluate(
          model,
          _eval_data,
          model_type="question-answering",
          evaluators="default",
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
   
if __name__ == "__main__":
    #mlflow.set_tracking_uri("http://localhost:8080") $ pr mlflow server --host localhost --port 8080
    #vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    evaluate_rag()

