# LangChain, LangGraph, LangSmith

Python RAG using LangChain, LangGraph and LangSmith with local memory checkpoints. It runs on Quart HTTP/3 ASGI framework.

## Environment

Add a `.env` with the following environment variables:

```
LANGSMITH_TRACING="true"
LANGSMITH_API_KEY=""
LANGSMITH_TRACING="true"
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_PROJECT=""
OPENAI_API_KEY=""
GOOGLE_CLOUD_PROJECT=""
GOOGLE_CLOUD_LOCATION="us-central1"
GEMINI_API_KEY=""
VERTEX_API_KEY=""
GOOGLE_GENAI_USE_VERTEXAI="true"
USER_AGENT="USER_AGENT"
```

- Install `tkinter`:

```
$ sudo apt install -y python3.13-tk
```

## Google VertexAI

- Install Google Cloud CLI:

```
$ pip3 install --upgrade google-cloud-aiplatform
$ sudo snap install google-cloud-cli --classic
```

- Setup Google Cloud Authentication and Project:

```
$ gcloud init
$ gcloud auth application-default login
$ gcloud auth application-default set-quota-project <ProjectID>
```

### Google account setup in Docker to run on k8s

- https://cloud.google.com/docs/authentication/set-up-adc-local-dev-environment
  (1) Create a service account:

```
$ gcloud iam service-accounts create <sa-name> --display-name=<display_name>
```

(2) Create service account keys

```
$ gcloud iam service-account keys create service-account.json --iam-account=<sa-name>@<project>.gserviceaccount.com
```

- To get FQDN of the service account:

```
$  gcloud iam service-accounts list
```

(3) Create a secret from the json file:

```
$ k create secret generic gcloud-service-account --from-file=service-account.json
```

## Launch LangGraph Server

### Checkpointed RAG

- Configure `langgraph.json` with:

```
    "graphs": {
        "rag_agent": "./src/rag_agent/CheckpointedRAG.py:make_graph"
    },

```

### RAG ReAct Agent

- Configure `langgraph.json` with:

```
    "graphs": {
        "rag_agent": "./src/rag_agent/RAGAgent.py:make_graph"
    },

```

### Run local Langgraph server

- `langgraph dev`

### StateGraph with Checkpoint

![StateGraph with Checkpoint](./checkpoint_graph.png?raw=true "StateGraph with Checkpoint")

```
$ p -m src.rag_agent.CheckpointedRAG
================================ Human Message =================================

What is Task Decomposition?
================================== Ai Message ==================================
Tool Calls:
  retrieve (7f55237f-1295-45a1-a264-50d7eeccf60e)
 Call ID: 7f55237f-1295-45a1-a264-50d7eeccf60e
  Args:
    query: What is Task Decomposition?
================================= Tool Message =================================
Name: retrieve

Source: {'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}
Content: Fig. 1. Overview of a LLM-powered autonomous agent system.
Component One: Planning#
A complicated task usually involves many steps. An agent needs to know what they are and plan ahead.
Task Decomposition#
Chain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.

Source: {'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}
Content: Tree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.
Task decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.
================================== Ai Message ==================================

Task decomposition is the process of breaking down a complex task into smaller, more manageable steps. This can be achieved through prompting techniques like Chain of Thought (CoT), which encourages the model to "think step by step." Task decomposition can be done by LLM with simple prompting, task-specific instructions or with human inputs.
================================ Human Message =================================

Can you look up some common ways of doing it?
================================== Ai Message ==================================
Tool Calls:
  retrieve (22c61c84-8cda-4d91-a453-1b645a354d50)
 Call ID: 22c61c84-8cda-4d91-a453-1b645a354d50
  Args:
    query: common ways to do task decomposition
================================= Tool Message =================================
Name: retrieve

Source: {'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}
Content: Fig. 1. Overview of a LLM-powered autonomous agent system.
Component One: Planning#
A complicated task usually involves many steps. An agent needs to know what they are and plan ahead.
Task Decomposition#
Chain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.

Source: {'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}
Content: Tree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.
Task decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.
```

### ReAct Agent with Checkpoint

![ReAct Agent with Checkpoint](./agent_graph.png?raw=true "ReAct Agent with Checkpoint")

```
$ p -m src.rag_agent.RAGAgent
USER_AGENT environment variable not set, consider setting it to identify your requests.

=== CreateGraph ===

=== LoadDocuments ===
Total characters: 43130

=== _SplitDocuments ===
Split blog post into 66 sub-documents.

=== _IndexChunks ===
66 documents added successfully!

=== ChatAgent ===
================================ Human Message =================================

['What is the standard method for Task Decomposition?', 'Once you get the answer, look up common extensions of that method.']
================================== Ai Message ==================================
Name: RAG ReAct Agent
Tool Calls:
  retrieve (485afded-5e53-4c71-8783-b90f6db287b7)
 Call ID: 485afded-5e53-4c71-8783-b90f6db287b7
  Args:
    query: standard method for Task Decomposition

=== asimilarity_search ===
Retrying vertexai.language_models._language_models._TextEmbeddingModel.get_embeddings in 4.0 seconds as it raised ResourceExhausted: 429 Quota exceeded for aiplatform.googleapis.com/online_prediction_requests_per_base_model with base model: textembedding-gecko. Please submit a quota increase request. https://cloud.google.com/vertex-ai/docs/generative-ai/quotas-genai..
Retrying vertexai.language_models._language_models._TextEmbeddingModel.get_embeddings in 4.0 seconds as it raised ResourceExhausted: 429 Quota exceeded for aiplatform.googleapis.com/online_prediction_requests_per_base_model with base model: textembedding-gecko. Please submit a quota increase request. https://cloud.google.com/vertex-ai/docs/generative-ai/quotas-genai..
================================= Tool Message =================================
Name: retrieve

Source: {'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}
Content: Fig. 1. Overview of a LLM-powered autonomous agent system.
Component One: Planning#
A complicated task usually involves many steps. An agent needs to know what they are and plan ahead.
Task Decomposition#
Chain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.

Source: {'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}
Content: (3) Task execution: Expert models execute on the specific tasks and log results.
Instruction:

With the input and the inference results, the AI assistant needs to describe the process and results. The previous stages can be formed as - User Input: {{ User Input }}, Task Planning: {{ Tasks }}, Model Selection: {{ Model Assignment }}, Task Execution: {{ Predictions }}. You must first answer the user's request in a straightforward manner. Then describe the task process and show your analysis and model inference results to the user in the first person. If inference results contain a file path, must tell the user the complete file path.
================================== Ai Message ==================================
Name: RAG ReAct Agent

Okay, I will strive to provide accurate answers. Based on the information I have, Chain of Thought (CoT) prompting is becoming a standard technique for task decomposition, where the model is instructed to "think step by step" to break down complex tasks into smaller, simpler steps.

Now I will search for common extensions of the Chain of Thought method.
Tool Calls:
  ground_search (6e940bcb-73df-4951-a3b7-18e1cb8c373d)
 Call ID: 6e940bcb-73df-4951-a3b7-18e1cb8c373d
  Args:
    query: common extensions of Chain of Thought prompting
================================= Tool Message =================================
Name: ground_search

Chain of Thought (CoT) prompting has evolved into various extensions and variations that aim to improve its performance, address specific challenges, and broaden its applicability. Here are some common extensions of Chain of Thought prompting:

*   **Zero-Shot CoT:** This approach leverages the inherent knowledge within models to tackle problems without requiring prior specific examples or fine-tuning. It typically involves adding the phrase "Let's think step by step" to the prompt.
*   **Automatic Chain of Thought (Auto-CoT):** This method automatically generates intermediate reasoning steps, further automating the prompting process. It uses techniques like clustering questions based on semantic similarity to ensure diverse reasoning patterns are covered.
*   **Contrastive Chain-of-Thought:** This extends the standard CoT by providing examples of both positive and negative answers in the context. This helps the model learn what mistakes to avoid, potentially leading to fewer errors.
*   **Multimodal CoT:** Traditional CoT focuses on the language modality. Multimodal CoT incorporates text and vision into a two-stage framework. The first step involves rationale generation based on multimodal information.
*   **Program of Thoughts (PoT):** In Chain-of-Thought (CoT) Prompting, LLMs perform both reasoning and computations. The LLM generates mathematical expressions as a reasoning step and then solves it to get the final answer. However, LLMs are not the ideal candidate for solving mathematical expressions as they are not capable of solving complex mathematical expressions and are inefficient for performing iterative numerical computations. Program of Thoughts (PoT) prompting technique delegates the computation steps to an external language interpreter such as a python to get accurate response.
*   **Tree of Thoughts (ToT):** ToT extends CoT by exploring multiple reasoning possibilities at each step. It decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS or DFS while each state is evaluated by a classifier (via a prompt) or majority vote.
*   **Graph of Thoughts (GoT):** This extension requires building a graph framework through LLMs. The GoT architecture includes a set of interacting modules consisting of a prompter, parser, scoring module, and controller.
*   **Self-Consistency:** This technique improves performance by sampling multiple, diverse chains of thought for the same problem and then selecting the most consistent answer from these chains.
*   **Active Prompting with Chain-of-Thought:** This involves actively selecting the most informative examples to include in the prompt, which can improve the model's performance and data efficiency.

These extensions demonstrate the ongoing research and development in the field of chain-of-thought prompting, with the goal of enhancing the reasoning and problem-solving capabilities of large language models.

================================== Ai Message ==================================
Name: RAG ReAct Agent

Okay, I will strive to provide accurate answers. Based on the information I have:

The standard method for task decomposition is Chain of Thought (CoT) prompting, where the model is instructed to "think step by step" to break down complex tasks into smaller, simpler steps.

Common extensions of Chain of Thought prompting include:

*   **Zero-Shot CoT:** Adding "Let's think step by step" to the prompt.
*   **Automatic Chain of Thought (Auto-CoT):** Automatically generates intermediate reasoning steps.
*   **Contrastive Chain-of-Thought:** Providing examples of both positive and negative answers.
*   **Multimodal CoT:** Incorporates text and vision.
*   **Program of Thoughts (PoT):** Delegates computation steps to an external language interpreter.
*   **Tree of Thoughts (ToT):** Explores multiple reasoning possibilities at each step, creating a tree structure.
*   **Graph of Thoughts (GoT):** Builds a graph framework.
*   **Self-Consistency:** Samples multiple chains of thought and selects the most consistent answer.
*   **Active Prompting with Chain-of-Thought:** Actively selects informative examples to include in the prompt.
```

## LangSmith Application trace

- https://smith.langchain.com/

## Diagnostics

- HTTP/3 curl:

```
$ docker run --rm ymuski/curl-http3 curl --http3 --verbose https://<nodeport service>:<nodeport>/healthz/ready
```

## Neo4J

### To import CSV into the database:

- Need to copy the files / folder into the pod `/var/lib/neo4j/import`

### Sample Cypher Queries:

- Visit with id:56:

```
MATCH (v:Visit) WHERE v.id = 56 RETURN v;
```

- Which patient was involved in Visit id:56

```
MATCH (p:Patient)-[h:HAS]->(v:Visit) where v.id=56 return v,h,p
```

- Which physician treated the patient i Visit id:56

```
MATCH (p:Patient)-[h:HAS]->(v:Visit)<-[t:TREATS]-(ph:Physician) where v.id=56 return v,h,p,t,ph
```

- All relationships going in and out of Visit id:56

```
MATCH (v:Visit)-[r]-(n) where v.id=56 return v,r,n
```

### Sample Cypher Accumulation:

- Total visits and bill paid by payer Aetna in Texas:

```
MATCH (p:Payer)<-[c:COVERED_BY]-(v:Visit)-[:AT]->(h:Hospital)
WHERE p.name = "Aetna"
AND h.state_name = "TX"
RETURN COUNT(*) as num_visits,
SUM(c.billing_amount) as total_billing_amount;
```
