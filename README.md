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
  retrieve (c53aee07-1702-4731-b9c4-f4822822db30)
 Call ID: c53aee07-1702-4731-b9c4-f4822822db30
  Args:
    query: standard method for Task Decomposition

=== asimilarity_search ===
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

Okay, I will make sure to provide accurate answers. Based on the information I have, Chain of Thought (CoT) is a standard prompting technique for task decomposition, where the model is instructed to "think step by step" to break down complex tasks into smaller, simpler steps.

Now I will look up common extensions of the Chain of Thought method.
Tool Calls:
  ground_search (81a3e74f-991b-48f2-b545-5956822be14d)
 Call ID: 81a3e74f-991b-48f2-b545-5956822be14d
  Args:
    query: extensions of Chain of Thought prompting
================================= Tool Message =================================
Name: ground_search

["Chain of Thought (CoT) prompting has spurred the development of several extensions and variations that aim to improve the reasoning capabilities of large language models (LLMs). Here are some notable extensions:\n\n*   **Chain of Thought with Self-Consistency (CoT-SC):** This technique generates multiple reasoning paths and then selects the most consistent answer or path.\n*   **Automatic Chain of Thought (Auto-CoT):** This approach automatically generates intermediate reasoning steps. It uses a procedure to generate reasoning chains for CoT prompting, extending the automation introduced by zero-shot prompting. To use Auto-CoT, you need to apply BERT or a similar model to embed and cluster questions based on semantic similarity to ensure the selected demonstrations cover a diverse range of reasoning patterns.\n*   **Contrastive Chain-of-Thought Prompting:** This extends the standard CoT by providing examples of both positive and negative answers in the context to guide the model in reasoning step-by-step while reducing mistakes.\n*   **Graph of Thoughts (GoT):** This involves building a graph framework through LLMs. The GoT architecture includes modules like a prompter, parser, scoring module, and controller to coordinate the reasoning process.\n*   **Program of Thoughts (PoT):** This technique delegates computation steps to an external language interpreter like Python to get accurate responses, as LLMs are not ideal for solving complex mathematical expressions or performing iterative numerical computations.\n*   **Tree of Thoughts:** This extends CoT by exploring multiple reasoning possibilities at each step, creating a tree structure. The search process can be BFS (Breadth-First Search) or DFS (Depth-First Search), with each state evaluated by a classifier or majority vote.\n*   **Augment-Prune-Select:** This three-step process automatically constructs chain-of-thought prompts:\n\n    *   **Augment:** Generate multiple pseudo-chains of thought given a question using few-shot or zero-shot CoT prompts.\n    *   **Prune:** Prune pseudo chains based on whether generated answers match ground truths.\n    *   **Select:** Apply a variance-reduced policy gradient strategy to learn the probability distribution over selected examples, considering the probability distribution over examples as policy and the validation set accuracy as reward.\n\nThese extensions often aim to improve accuracy, reduce errors, and enhance the transparency and interpretability of the reasoning process in LLMs.\n"]
================================== Ai Message ==================================
Name: RAG ReAct Agent

Okay, I apologize for any previous inaccuracies. Based on my search, here are some common extensions of Chain of Thought (CoT) prompting:

*   **Chain of Thought with Self-Consistency (CoT-SC):** Generates multiple reasoning paths and selects the most consistent answer.
*   **Automatic Chain of Thought (Auto-CoT):** Automatically generates intermediate reasoning steps using semantic similarity clustering.
*   **Contrastive Chain-of-Thought Prompting:** Provides examples of both positive and negative answers to guide reasoning and reduce mistakes.
*   **Graph of Thoughts (GoT):** Builds a graph framework to coordinate the reasoning process.
*   **Program of Thoughts (PoT):** Delegates computation steps to an external language interpreter like Python.
*   **Tree of Thoughts:** Explores multiple reasoning possibilities at each step in a tree structure.
*   **Augment-Prune-Select:** Automatically constructs chain-of-thought prompts through augmentation, pruning, and selection.

These extensions aim to improve accuracy, reduce errors, and enhance the transparency and interpretability of the reasoning process in large language models.
```

## LangSmith Application trace

- https://smith.langchain.com/
