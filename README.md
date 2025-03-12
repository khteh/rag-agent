# LangChain, LangGraph, LangSmith

Python RAG using LangChain, LangGraph and LangSmith with local memory checkpoints

## Environment

Add a `.env` with the following environment variables:

```
LANGSMITH_TRACING="true"
LANGSMITH_API_KEY="<foo>"
LANGSMITH_TRACING="true"
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="<foo>"
LANGSMITH_PROJECT="<foo>"
OPENAI_API_KEY="<foo>"
VERTEXAI_PROJECT_ID="<foo>"
VERTEXAI_PROJECT_LOCATION="<foo>"
```

## Google VertexAI

- Install Google Cloud CLI:

```
$ pip3 install --upgrade google-cloud-aiplatform
$ sudo snap install google-cloud-cli --classic
```

- Setup Google Cloud Authentication and Project:

```
$ gcloud auth application-default login
$ gcloud auth application-default set-quota-project <ProjectID>
```

## Launch LangGraph Server

- `langgraph dev`

### StateGraph with Checkpoint

![StateGraph with Checkpoint](./checkpoint_graph.png?raw=true "StateGraph with Checkpoint")

```
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
================================ Human Message =================================

['What is the standard method for Task Decomposition?', 'Once you get the answer, look up common extensions of that method.']
================================== Ai Message ==================================
Name: RAG ReAct Agent
Tool Calls:
  retrieve (8c1c7734-5f43-4c75-b9f3-d1fcb4c8a1a7)
 Call ID: 8c1c7734-5f43-4c75-b9f3-d1fcb4c8a1a7
  Args:
    query: standard method for Task Decomposition
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

Okay, I will make sure to provide accurate answers. Based on the information I found, Chain of Thought (CoT) is becoming a standard prompting technique for task decomposition, where a model is instructed to "think step by step" to break down complex tasks into smaller, simpler steps.

Now I will look up common extensions of the Chain of Thought method.
Tool Calls:
  retrieve (ff990bb9-015f-4ef1-9319-6d72f21d3ad9)
 Call ID: ff990bb9-015f-4ef1-9319-6d72f21d3ad9
  Args:
    query: common extensions of Chain of Thought prompting
================================= Tool Message =================================
Name: retrieve

Source: {'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}
Content: Tree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.
Task decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.

Source: {'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}
Content: Fig. 1. Overview of a LLM-powered autonomous agent system.
Component One: Planning#
A complicated task usually involves many steps. An agent needs to know what they are and plan ahead.
Task Decomposition#
Chain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.
================================== Ai Message ==================================
Name: RAG ReAct Agent

Okay, I will make sure to provide accurate answers. Based on the information I found, Tree of Thoughts is a common extension of Chain of Thought. Tree of Thoughts explores multiple reasoning possibilities at each step, decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote. Task decomposition can also be done by LLM with simple prompting, by using task-specific instructions, or with human inputs.
```

## LangSmith Application trace

- https://smith.langchain.com/
