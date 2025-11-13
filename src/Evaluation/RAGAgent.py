import mlflow, pandas
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
#from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
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
  }
)
def model(input_df):
    # For VertexAI, use VertexAIEmbeddings, model="text-embedding-005"; "gemini-2.0-flash" model_provider="google_genai"
    model = init_chat_model(config.LLM_RAG_MODEL, model_provider="ollama", base_url=config.OLLAMA_URI, streaming=True, temperature=0)
    vectorStore = VectorStore(model=config.EMBEDDING_MODEL, chunk_size=1000, chunk_overlap=0)
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
    relevance_metric = relevance(
      model="endpoints:/databricks-llama-2-70b-chat"
    )  # You can also use any model you have hosted on Databricks, models from the Marketplace or models in the Foundation model API
    mlflow.set_experiment(evaluate_rag.__name__)
    mlflow.langchain.autolog()
  
    with mlflow.start_run():
      results = mlflow.genai.evaluate(
          model,
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
   
if __name__ == "__main__":
    #mlflow.set_tracking_uri("http://localhost:8080") $ pr mlflow server --host localhost --port 8080
    #vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    evaluate_rag()

