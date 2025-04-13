# LangChain, LangGraph, LangSmith

Python RAG using LangChain, LangGraph and LangSmith with local memory checkpoints. It runs on Quart/Hypercorn HTTP/3 ASGI framework.

## Environment

Add a `.env` with the following environment variables:

```
DB_USERNAME=
DB_PASSWORD=
NEO4J_AUTH=username/password
CHROMA_TOKEN=
LANGSMITH_TRACING="true"
LANGSMITH_API_KEY=
LANGSMITH_TRACING="true"
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_PROJECT=
OPENAI_API_KEY=
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=
GOOGLE_CLOUD_LOCATION="asia-southeast1"
GOOGLE_API_KEY=
GEMINI_API_KEY=
VERTEX_API_KEY=
GOOGLE_GENAI_USE_VERTEXAI="true"
USER_AGENT="USER_AGENT"
```

- Install `tkinter`:

```
$ sudo apt install -y python3.13-tk
```

## Local Model

- https://python.langchain.com/docs/how_to/local_llms/

### Ollama

- https://github.com/ollama/ollama
- Download and install the app:

```
$ curl -fsSL https://ollama.com/install.sh | sh
```

- Run the app with a model:

```
$ ollama run llama3.3
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

## Start the application:

- `./hypercorn.sh`

## Curl

- Add `Host` header which is defined as `server_names` in `hypercorn.toml`

## Chrome browser

- Close ALL chrome browser (both tabs and windows)
- Generate TLS certificate and it's fingerprint:

```
$ fingerprint=`openssl x509 -pubkey -noout -in /tmp/server.crt |
        openssl rsa -pubin -outform der |
        openssl dgst -sha256 -binary | base64`
```

- Start Chrome browser with QUIC protocol for HTTP/3:
- `$1` is URL. For example: `https://localhost:4433`

```
$  /opt/google/chrome/chrome --disable-setuid-sandbox --enable-quic --ignore-certificate-errors-spki-list=$fingerprint --origin-to-force-quic-on=${1#*//} $1
```

### Home controller endpoints:

```
$ c3 -v https://localhost:4433/invoke -m 300 -X POST -d '{"message": "What is task decomposition?"}'
```

### Hospital controller endpoints:

```
$ c3 -v https://localhost:4433/healthcare/invoke -m 300 -X POST -d '{"message": "Which hospital has the shortest wait time?"}'
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

### ReAct Agent with Checkpoint

![ReAct Agent with Checkpoint](./agent_graph.png?raw=true "ReAct Agent with Checkpoint")
![ReAct Agent UI](./rag-agent.png?raw=true "ReAct Agent UI")

```
$ p -m src.rag_agent.RAGAgent
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

### Email RAG StateGraph with Checkpoint

![Email RAG StateGraph with Checkpoint](./EmailRAGStateGraph.png?raw=true "Email RAG StateGraph with Checkpoint")
![Email RAG Agent with Checkpoint](./EmailRAGAgent.png?raw=true "Email RAG Agent with Checkpoint")

```
$ pp src.rag_agent.EmailRAG
2025-04-02 18:22:31 INFO
=== __init__ ===
2025-04-02 18:22:31 INFO
=== CreateGraph ===
2025-04-02 18:22:31 DEBUG    Starting new HTTPS connection (1): mermaid.ink:443
2025-04-02 18:22:32 DEBUG    https://mermaid.ink:443 "GET /img/LS0tCmNvbmZpZzoKICBmbG93Y2hhcnQ6CiAgICBjdXJ2ZTogbGluZWFyCi0tLQpncmFwaCBURDsKCV9fc3RhcnRfXyhbPHA+X19zdGFydF9fPC9wPl0pOjo6Zmlyc3QKCVBhcnNlRW1haWwoUGFyc2VFbWFpbCkKCU5lZWRzRXNjYWxhdGlvbihOZWVkc0VzY2FsYXRpb24pCglfX2VuZF9fKFs8cD5fX2VuZF9fPC9wPl0pOjo6bGFzdAoJTmVlZHNFc2NhbGF0aW9uIC0tPiBfX2VuZF9fOwoJUGFyc2VFbWFpbCAtLT4gTmVlZHNFc2NhbGF0aW9uOwoJX19zdGFydF9fIC0tPiBQYXJzZUVtYWlsOwoJY2xhc3NEZWYgZGVmYXVsdCBmaWxsOiNmMmYwZmYsbGluZS1oZWlnaHQ6MS4yCgljbGFzc0RlZiBmaXJzdCBmaWxsLW9wYWNpdHk6MAoJY2xhc3NEZWYgbGFzdCBmaWxsOiNiZmI2ZmMK?type=png&bgColor=!white HTTP/1.1" 200 9617
2025-04-02 18:22:32 DEBUG    STREAM b'IHDR' 16 13
2025-04-02 18:22:32 DEBUG    STREAM b'sRGB' 41 1
2025-04-02 18:22:32 DEBUG    STREAM b'IDAT' 54 8192
2025-04-02 18:22:35 DEBUG    Starting new HTTPS connection (1): mermaid.ink:443
2025-04-02 18:22:35 DEBUG    https://mermaid.ink:443 "GET /img/LS0tCmNvbmZpZzoKICBmbG93Y2hhcnQ6CiAgICBjdXJ2ZTogbGluZWFyCi0tLQpncmFwaCBURDsKCV9fc3RhcnRfXyhbPHA+X19zdGFydF9fPC9wPl0pOjo6Zmlyc3QKCUVtYWlsQWdlbnQoRW1haWxBZ2VudCkKCUVtYWlsVG9vbHMoRW1haWxUb29scykKCV9fZW5kX18oWzxwPl9fZW5kX188L3A+XSk6OjpsYXN0CglFbWFpbFRvb2xzIC0tPiBFbWFpbEFnZW50OwoJX19zdGFydF9fIC0tPiBFbWFpbEFnZW50OwoJRW1haWxBZ2VudCAtLi0+IEVtYWlsVG9vbHM7CglFbWFpbEFnZW50IC0uLT4gX19lbmRfXzsKCWNsYXNzRGVmIGRlZmF1bHQgZmlsbDojZjJmMGZmLGxpbmUtaGVpZ2h0OjEuMgoJY2xhc3NEZWYgZmlyc3QgZmlsbC1vcGFjaXR5OjAKCWNsYXNzRGVmIGxhc3QgZmlsbDojYmZiNmZjCg==?type=png&bgColor=!white HTTP/1.1" 200 10010
2025-04-02 18:22:35 DEBUG    STREAM b'IHDR' 16 13
2025-04-02 18:22:35 DEBUG    STREAM b'sRGB' 41 1
2025-04-02 18:22:35 DEBUG    STREAM b'IDAT' 54 8192
2025-04-02 18:22:36 INFO
=== Chat ===
2025-04-02 18:22:36 DEBUG    Starting new HTTPS connection (1): api.smith.langchain.com:443
================================ Human Message =================================

The escalation criteria is: "There's an immediate risk of electrical, water, or fire damage". Here's the email: "
    Date: January 10, 2025
    From: City of Los Angeles Building and Safety Department
    To: West Coast Development, project 345678123 - Sunset Luxury
    Condominiums
    Location: Los Angeles, CA
    Following an inspection of your site at 456 Sunset Boulevard, we have
    identified the following building code violations:
    Electrical Wiring: Exposed wiring was found in the underground parking
    garage, posing a safety hazard. Fire Safety: Insufficient fire
    extinguishers were available across multiple floors of the structure
    under construction.
    Structural Integrity: The temporary support beams in the eastern wing
    do not meet the load-bearing standards specified in local building
    codes.
    Required Corrective Actions:
    Replace or properly secure exposed wiring to meet electrical safety
    standards. Install additional fire extinguishers in compliance with
    fire code requirements. Reinforce or replace temporary support beams
    to ensure structural stability. Deadline for Compliance: Violations
    must be addressed no later than February 5,
    2025. Failure to comply may result in
    a stop-work order and additional fines.
    Contact: For questions or to schedule a re-inspection, please contact
    the Building and Safety Department at
    (555) 456-7890 or email inspections@lacity.gov.
    "
2025-04-02 18:22:36 DEBUG    connect_tcp.started host='127.0.0.1' port=11434 local_address=None timeout=None socket_options=None
2025-04-02 18:22:36 DEBUG    connect_tcp.complete return_value=<httpcore._backends.anyio.AnyIOStream object at 0x7ff93ca387d0>
2025-04-02 18:22:36 DEBUG    send_request_headers.started request=<Request [b'POST']>
2025-04-02 18:22:36 DEBUG    send_request_headers.complete
2025-04-02 18:22:36 DEBUG    send_request_body.started request=<Request [b'POST']>
2025-04-02 18:22:36 DEBUG    send_request_body.complete
2025-04-02 18:22:36 DEBUG    receive_response_headers.started request=<Request [b'POST']>
2025-04-02 18:22:37 DEBUG    https://api.smith.langchain.com:443 "GET /info HTTP/1.1" 200 672
2025-04-02 18:22:37 DEBUG    https://api.smith.langchain.com:443 "POST /runs/multipart HTTP/1.1" 202 34
2025-04-02 18:22:40 DEBUG    receive_response_headers.complete return_value=(b'HTTP/1.1', 200, b'OK', [(b'Content-Type', b'application/x-ndjson'), (b'Date', b'Wed, 02 Apr 2025 10:22:40 GMT'), (b'Transfer-Encoding', b'chunked')])
2025-04-02 18:22:40 DEBUG    receive_response_body.started request=<Request [b'POST']>
2025-04-02 18:22:40 DEBUG    receive_response_body.complete
2025-04-02 18:22:40 DEBUG    response_closed.started
2025-04-02 18:22:40 DEBUG    response_closed.complete
================================== Ai Message ==================================
Tool Calls:
  email_processing_tool (edf51167-3128-4083-9513-d4d7850c6878)
 Call ID: edf51167-3128-4083-9513-d4d7850c6878
  Args:
    email: Following an inspection of your site at 456 Sunset Boulevard, we have identified the following building code violations: Electrical Wiring: Exposed wiring was found in the underground parking garage, posing a safety hazard. Fire Safety: Insufficient fire extinguishers were available across multiple floors of the structure under construction. Structural Integrity: The temporary support beams in the eastern wing do not meet the load-bearing standards specified in local building codes.
    escalation_criteria: There
2025-04-02 18:22:40 INFO
=== email_processing_tool ===
2025-04-02 18:22:40 DEBUG    email: Following an inspection of your site at 456 Sunset Boulevard, we have identified the following building code violations: Electrical Wiring: Exposed wiring was found in the underground parking garage, posing a safety hazard. Fire Safety: Insufficient fire extinguishers were available across multiple floors of the structure under construction. Structural Integrity: The temporary support beams in the eastern wing do not meet the load-bearing standards specified in local building codes., escalation_criteria: There, emailState:: {'escalation_dollar_criteria': 100000, 'escalation_emails': ['brog@abc.com', 'bigceo@company.com'], 'email': 'Following an inspection of your site at 456 Sunset Boulevard, we have identified the following building code violations: Electrical Wiring: Exposed wiring was found in the underground parking garage, posing a safety hazard. Fire Safety: Insufficient fire extinguishers were available across multiple floors of the structure under construction. Structural Integrity: The temporary support beams in the eastern wing do not meet the load-bearing standards specified in local building codes.', 'escalation_text_criteria': 'There'}
2025-04-02 18:22:40 INFO
=== ParseEmail ===
2025-04-02 18:22:40 DEBUG    state: {'email': 'Following an inspection of your site at 456 Sunset Boulevard, we have identified the following building code violations: Electrical Wiring: Exposed wiring was found in the underground parking garage, posing a safety hazard. Fire Safety: Insufficient fire extinguishers were available across multiple floors of the structure under construction. Structural Integrity: The temporary support beams in the eastern wing do not meet the load-bearing standards specified in local building codes.', 'escalation_text_criteria': 'There', 'escalation_dollar_criteria': 100000, 'escalation_emails': ['brog@abc.com', 'bigceo@company.com']}
2025-04-02 18:22:40 DEBUG    connect_tcp.started host='127.0.0.1' port=11434 local_address=None timeout=None socket_options=None
2025-04-02 18:22:40 DEBUG    connect_tcp.complete return_value=<httpcore._backends.anyio.AnyIOStream object at 0x7ff93cab2210>
2025-04-02 18:22:40 DEBUG    send_request_headers.started request=<Request [b'POST']>
2025-04-02 18:22:40 DEBUG    send_request_headers.complete
2025-04-02 18:22:40 DEBUG    send_request_body.started request=<Request [b'POST']>
2025-04-02 18:22:40 DEBUG    send_request_body.complete
2025-04-02 18:22:40 DEBUG    receive_response_headers.started request=<Request [b'POST']>
2025-04-02 18:22:40 DEBUG    receive_response_headers.complete return_value=(b'HTTP/1.1', 200, b'OK', [(b'Content-Type', b'application/x-ndjson'), (b'Date', b'Wed, 02 Apr 2025 10:22:40 GMT'), (b'Transfer-Encoding', b'chunked')])
2025-04-02 18:22:40 DEBUG    receive_response_body.started request=<Request [b'POST']>
2025-04-02 18:22:41 DEBUG    https://api.smith.langchain.com:443 "POST /runs/multipart HTTP/1.1" 202 34
2025-04-02 18:22:42 DEBUG    receive_response_body.complete
2025-04-02 18:22:42 DEBUG    response_closed.started
2025-04-02 18:22:42 DEBUG    response_closed.complete
2025-04-02 18:22:42 DEBUG    Extract: name=None phone=None email=None project_id=1234 site_location=None violation_type='Electrical Wiring' required_changes='Replace with approved conduit and secure all exposed wiring.' max_potential_fine=250000.0 date_of_notice=None compliance_deadline=None
2025-04-02 18:22:42 INFO
=== NeedsEscalation ===
2025-04-02 18:22:42 DEBUG    state: {'email': 'Following an inspection of your site at 456 Sunset Boulevard, we have identified the following building code violations: Electrical Wiring: Exposed wiring was found in the underground parking garage, posing a safety hazard. Fire Safety: Insufficient fire extinguishers were available across multiple floors of the structure under construction. Structural Integrity: The temporary support beams in the eastern wing do not meet the load-bearing standards specified in local building codes.', 'extract': EmailModel(name=None, phone=None, email=None, project_id=1234, site_location=None, violation_type='Electrical Wiring', required_changes='Replace with approved conduit and secure all exposed wiring.', max_potential_fine=250000.0, date_of_notice=None, compliance_deadline=None), 'escalation_text_criteria': 'There', 'escalation_dollar_criteria': 100000, 'escalation_emails': ['brog@abc.com', 'bigceo@company.com']}
2025-04-02 18:22:42 DEBUG    connect_tcp.started host='127.0.0.1' port=11434 local_address=None timeout=None socket_options=None
2025-04-02 18:22:42 DEBUG    connect_tcp.complete return_value=<httpcore._backends.anyio.AnyIOStream object at 0x7ff93c440ec0>
2025-04-02 18:22:42 DEBUG    send_request_headers.started request=<Request [b'POST']>
2025-04-02 18:22:42 DEBUG    send_request_headers.complete
2025-04-02 18:22:42 DEBUG    send_request_body.started request=<Request [b'POST']>
2025-04-02 18:22:42 DEBUG    send_request_body.complete
2025-04-02 18:22:42 DEBUG    receive_response_headers.started request=<Request [b'POST']>
2025-04-02 18:22:42 DEBUG    receive_response_headers.complete return_value=(b'HTTP/1.1', 200, b'OK', [(b'Content-Type', b'application/x-ndjson'), (b'Date', b'Wed, 02 Apr 2025 10:22:42 GMT'), (b'Transfer-Encoding', b'chunked')])
2025-04-02 18:22:42 DEBUG    receive_response_body.started request=<Request [b'POST']>
2025-04-02 18:22:42 DEBUG    receive_response_body.complete
2025-04-02 18:22:42 DEBUG    response_closed.started
2025-04-02 18:22:42 DEBUG    response_closed.complete
2025-04-02 18:22:42 DEBUG    result: needs_escalation=True
2025-04-02 18:22:42 DEBUG    result: {'email': 'Following an inspection of your site at 456 Sunset Boulevard, we have identified the following building code violations: Electrical Wiring: Exposed wiring was found in the underground parking garage, posing a safety hazard. Fire Safety: Insufficient fire extinguishers were available across multiple floors of the structure under construction. Structural Integrity: The temporary support beams in the eastern wing do not meet the load-bearing standards specified in local building codes.', 'extract': EmailModel(name=None, phone=None, email=None, project_id=1234, site_location=None, violation_type='Electrical Wiring', required_changes='Replace with approved conduit and secure all exposed wiring.', max_potential_fine=250000.0, date_of_notice=None, compliance_deadline=None), 'escalation_text_criteria': 'There', 'escalation_dollar_criteria': 100000, 'escalate': True, 'escalation_emails': ['brog@abc.com', 'bigceo@company.com']}
strptime() argument 1 must be str, not None
unconverted data remains: T16:30:00
================================= Tool Message =================================
Name: email_processing_tool

name=None phone=None email=None project_id=1234 site_location=None violation_type='Electrical Wiring' required_changes='Replace with approved conduit and secure all exposed wiring.' max_potential_fine=250000.0 date_of_notice=None compliance_deadline=None
2025-04-02 18:22:42 DEBUG    connect_tcp.started host='127.0.0.1' port=11434 local_address=None timeout=None socket_options=None
2025-04-02 18:22:42 DEBUG    connect_tcp.complete return_value=<httpcore._backends.anyio.AnyIOStream object at 0x7ff93c9e6150>
2025-04-02 18:22:42 DEBUG    send_request_headers.started request=<Request [b'POST']>
2025-04-02 18:22:42 DEBUG    send_request_headers.complete
2025-04-02 18:22:42 DEBUG    send_request_body.started request=<Request [b'POST']>
2025-04-02 18:22:42 DEBUG    send_request_body.complete
2025-04-02 18:22:42 DEBUG    receive_response_headers.started request=<Request [b'POST']>
2025-04-02 18:22:43 DEBUG    https://api.smith.langchain.com:443 "POST /runs/multipart HTTP/1.1" 202 34
2025-04-02 18:22:45 DEBUG    receive_response_headers.complete return_value=(b'HTTP/1.1', 200, b'OK', [(b'Content-Type', b'application/x-ndjson'), (b'Date', b'Wed, 02 Apr 2025 10:22:45 GMT'), (b'Transfer-Encoding', b'chunked')])
2025-04-02 18:22:45 DEBUG    receive_response_body.started request=<Request [b'POST']>
2025-04-02 18:22:45 DEBUG    receive_response_body.complete
2025-04-02 18:22:45 DEBUG    response_closed.started
2025-04-02 18:22:45 DEBUG    response_closed.complete
================================== Ai Message ==================================

The situation at the Sunset Luxury Condominiums project (Project ID: 345678123) poses an immediate risk of electrical, water, or fire damage due to building code violations identified during a recent inspection.

To address these concerns, the following actions must be taken:

* Replace or properly secure exposed wiring in the underground parking garage to meet electrical safety standards.
* Install additional fire extinguishers across multiple floors of the structure in compliance with fire code requirements.
* Reinforce or replace temporary support beams in the eastern wing to ensure structural stability.

These corrective actions must be completed by February 5, 2025. Failure to comply may result in a stop-work order and additional fines.

If you have any questions or need to schedule a re-inspection, please contact the City of Los Angeles Building and Safety Department at (555) 456-7890 or email inspections@lacity.gov.
2025-04-02 18:22:46 DEBUG    https://api.smith.langchain.com:443 "POST /runs/multipart HTTP/1.1" 202 34
2025-04-02 18:22:46 DEBUG    Closing Client.session
2025-04-02 18:22:46 DEBUG    Closing Client.session
```

## LangSmith Application trace

- https://smith.langchain.com/

## Diagnostics

- HTTP/3 curl:

```
$ docker run --rm ymuski/curl-http3 curl --http3 --verbose https://<nodeport service>:<nodeport>/healthz/ready
```

- To build your own HTTP/3 curl: https://curl.se/docs/http3.html

## Neo4J

### To import CSV into the database:

- Need to copy the files / folder into the pod `/var/lib/neo4j/import`
- `LoadNeo4J.sh` will load `data/Healthcare` into `neo4j-0` pod

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
