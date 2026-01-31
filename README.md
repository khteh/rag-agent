# LLM-RAG Deep Agent using LangChain, LangGraph, LangSmith

Python LLM-RAG deep agent using LangChain, LangGraph and LangSmith built on Quart web microframework and served using Hypercorn ASGI and WSGI web server.

## Infrastructure components:

- All of the following components run on k8s cluster:

1. PostgreSQL for checkpoints and vector DB
2. (Optional) Chroma for vector DB
3. Neo4J for graph query
4. Ollama as LLM model server

## Sources of information for RAG

### Unstructured

- Online blog posts ingested into vector database
- Text strings extracted from SQL database ingested into Neo4J vector database
- Google ground search

### Structured

- SQL database relationships ingested into Neo4J graph database

## Environment

Add a `.env` with the following environment variables:

```
ENVIRONMENT=development
DB_USERNAME=
DB_PASSWORD=
NEO4J_AUTH=username/password
CHROMA_TOKEN=
LANGSMITH_TRACING="true"
LANGSMITH_API_KEY=
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_PROJECT=
OPENAI_API_KEY=
GOOGLE_CLOUD_PROJECT=
GOOGLE_CLOUD_LOCATION="asia-southeast1"
GOOGLE_API_KEY=
GEMINI_API_KEY=
VERTEX_API_KEY=
USER_AGENT="USER_AGENT"
```

- Install `tkinter`:

```
$ sudo apt install -y python3.13-tk
```

## Data Source Preparation:

### PostgreSQL Database

- Firstly, create an empty database "Langchain" in PostgreSQL

### Database Migration

- Copy `env.py` to `migrations/` folder.
- Set the values `DB_foo` in `/etc/ragagent_config.json`
- run migrations initialization with db init command:

  ```
  $ uv run alembic init migrations
  $ cp env.py migrations
  $ uv run alembic revision --autogenerate -m "Initial migration"
  $ uv run alembic upgrade head
  ```

- There will be 1 table, "users" created in the PostgreSQL database "Langchain" after the `upgrade`.

### PostgresSQL PGVector

- Run the following CLI command to crawl and ingest blogs into PostgreSQL PGVector.

```
$ uv run python -m src.rag_agent.RAGAgent -l
```

### Neo4J Graph DB

- Import data from CSV files into the database
- Copy the files / folder into the pod `/var/lib/neo4j/import`
- `LoadNeo4J.sh` will load `data/Healthcare` into `neo4j-0` pod

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
$ ollama run gpt-oss
```

## Google VertexAI / GenAI

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

## Start the application:

- To run the agent from console:

```
$ uv run python -m src.rag_agent.RAGAgent -h
usage: RAGAgent.py [-h] [-l] [-t] [-d] [-a] [-m] [-n] [-u] [-v] [-b] [-w]

LLM-RAG deep agent answering user questions about healthcare system and AI/ML

options:
  -h, --help            show this help message and exit
  -l, --load-urls       Load documents from URLs
  -t, --show-threads    Show history of all threads
  -d, --delete-threads  Delete history of all threads
  -a, --ai-ml           Ask questions regarding AI/ML which should be answered based on Lilian's blog
  -m, --mlflow          Ask questions regarding MLFlow
  -n, --neo4j-graph     Ask question with answer in Neo4J graph database store
  -u, --stream-updates  Stream updates instead of theh complete message
  -v, --neo4j-vector    Ask question with answers in Neo4J vector store
  -b, --neo4j           Ask question with answers in both Neo4J vector and graph stores
  -w, --wait-time       Ask hospital waiting time using answer from mock API endpoint
```

- To run the ASGI/WSGI web application:

```
$ ./hypercorn.sh
```

## HTTP/3 curl:

- To build your own HTTP/3 curl: https://curl.se/docs/http3.html
- Add `Host` header which is defined as `server_names` in `hypercorn.toml`

```
curl --http3-only --insecure -v https://localhost:4433/<path> -H "Host: khteh.com"
```

```
curl --http3-only --insecure -vv https://localhost:4433/chat/invoke -F 'prompt=:"What is task decomposition?"' -F 'file=@~/data/1.jpg' -F 'receipt=true'
```

## TLS certificates

- If run locally, `./hypercorn.sh` would have generated the TLS cert into `/tmp` folder
- If trying to access the application running in k8s cluster, copy the cert out of the pod into `/tmp` folder.

## Chrome browser

- Close ALL chrome browser (both tabs and windows)

- Generate TLS certificate fingerprint:

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

## Home controller endpoints:

```
$ c3 -v https://localhost:4433/invoke -m 300 -X POST -d '{"message": "What is task decomposition?"}'
```

## Hospital controller endpoints:

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

- `uv run python -m langgraph_cli dev`

## Main Agent:

![Main Agent](images/MainAgent.png?raw=true "Main Agent")

## Question & Answer RAG Deep Agent

### Sub-Agent

![Sub-Agent](images/agent_graph.png?raw=true "Sub-Agent")

### RAG Deep Agent answers question from Vector and Graph Database

![ReAct Agent UI](images/rag-agent.png?raw=true "ReAct Agent UI")

### Answering questions from mocked-up API call:

- `output/user_questions.md`:

```
Which hospital has the shortest wait time?
```

- `output/final_answer.md`:

```
## Shortest Wait Time Hospital

| Hospital | Location | Wait Time |
|----------|----------|-----------|
| **Walton LLC** | Florida (FL) | **6 minutes** |

**Sources**

- Shortest wait time: *get_most_available_hospital* returned “Walton LLC” with a wait time of 6 minutes.
- Location: *HealthcareCypher* confirmed that Walton LLC is located in Florida.
```

### Answering questions from Postgres Vector Store:

- `output/user_questions.md`:

```
What is task decomposition?
What is the standard method for Task Decomposition?
Once you get the answer, look up common extensions of that method.
```

- `output/final_answer.md`:

```
## Task Decomposition Overview

Task decomposition is the process of breaking a complex goal or problem into smaller, more manageable sub‑tasks that can be tackled sequentially or in parallel. This technique is fundamental in both human problem‑solving and in the design of autonomous agents and large language model (LLM) pipelines.

## Standard Method: Chain of Thought (CoT)

The most widely adopted standard for task decomposition in LLMs is the **Chain of Thought (CoT)** prompting strategy. In CoT, the model is explicitly asked to *think step by step* before producing a final answer. This encourages the generation of an intermediate reasoning chain that naturally decomposes the problem into a linear sequence of sub‑steps.

- **Prompt example**: "Think step by step to solve X."
- **Result**: A linear list of sub‑tasks that the model can execute or reason about.

CoT has become the baseline for many downstream techniques because it is simple to implement and works well across a wide range of domains.

## Common Extensions of CoT

| Extension | How it Builds on CoT | Typical Use‑Case |
|-----------|----------------------|------------------|
| **Tree of Thoughts (ToT)** | Generates multiple possible next steps at each node, forming a tree that can be searched with BFS/DFS. | Complex reasoning where multiple reasoning paths need exploration, such as puzzle solving or creative writing. |
| **LLM + Planner (LLM+P)** | The LLM translates the problem into a PDDL description, a classical planner generates a plan, and the LLM converts the plan back to natural language. | Long‑horizon planning tasks in robotics, logistics, or any domain with a formal planning representation. |
| **Tool‑Augmented Decomposition** | The LLM delegates sub‑tasks that exceed its internal knowledge to external tools (calculators, APIs, databases). | Tasks that require precise arithmetic, up‑to‑date data, or specialized computations. |
| **Human‑in‑the‑Loop** | A human reviews, refines, or reorders the decomposed steps. | Domains where domain expertise is critical or where the LLM’s confidence is low. |

## Practical Take‑aways

1. **CoT** remains the simplest and most widely used standard for decomposing tasks in LLMs.
2. **ToT** adds breadth, allowing exploration of alternative reasoning paths.
3. **LLM+P** is powerful for domains where a formal planner exists.
4. **Tool‑augmented** approaches extend the LLM’s reach beyond its internal knowledge base.

These methods collectively provide a toolkit for turning a single, complex instruction into a sequence of actionable sub‑tasks that an LLM or an autonomous agent can execute.

### Sources
[1] Task decomposition blog post – discusses CoT, ToT, LLM+P, and tool use.
```

- `output/user_questions.md`:

```
What is MLFLow?
```

- `output/final_answer.md`:

```
## MLFlow – A Quick‑Reference Guide

### Definition
MLFlow is an **open‑source platform** designed to manage the entire machine‑learning lifecycle—from experiment tracking to model deployment—while keeping everything reproducible and vendor‑neutral. It provides a set of APIs and UI tools that let data scientists and ML engineers log parameters, metrics, artifacts, and models in a single, searchable repository.

### Purpose
MLFlow’s core mission is to **reduce the operational friction** that comes with building, training, and deploying ML models. By centralizing experiment metadata and model artifacts, it enables teams to:

1. **Track and compare experiments** across different runs, libraries, and environments.
2. **Version models** and manage their lifecycle (staging → production → archived).
3. **Deploy models** to a variety of targets (REST APIs, cloud services, edge devices) with minimal boilerplate.
4. **Reproduce results** by storing all inputs, code, and environment details.

### Typical Usage
A typical MLFlow workflow looks like this:

1. **Experiment Tracking** – Log hyper‑parameters, metrics, and artifacts during training.
2. **Model Registry** – Register the best‑performing model, assign stages, and keep a lineage history.
3. **Deployment** – Serve the registered model via MLFlow’s built‑in REST API or push it to a cloud platform.
4. **Monitoring & Evaluation** – Continuously log new predictions and compare them against the original training metrics.

MLFlow works with popular libraries such as scikit‑learn, TensorFlow, PyTorch, and Spark MLlib, and can run locally, on‑prem, or in the cloud.

### Key Features

| Feature | What It Does | Benefit |
|---------|--------------|---------|
| **Experiment Tracking** | Logs parameters, metrics, and artifacts per run. | Easy comparison and visualization of model performance. |
| **Model Registry** | Centralized versioning, stage management, and lineage tracking. | Clear audit trail and controlled promotion to production. |
| **Model Deployment** | Supports REST API serving, batch inference, and cloud/edge deployment. | One‑click deployment to production environments. |
| **Dataset Tracking** | `mlflow.log_input()` records dataset metadata. | Ensures reproducibility by linking models to exact data versions. |
| **Integrations** | Autologging for scikit‑learn, TensorFlow, PyTorch, Spark, etc. | Minimal code changes to start logging. |
| **Observability & Evaluation** | Built‑in UI for visualizing metrics, plots, and model lineage. | Faster debugging and model quality assessment. |

### Sources
[1] MLflow: A Tool for Managing the Machine Learning Lifecycle – https://mlflow.org/docs/latest/index.html
[2] MLflow Tracking & Experiments – https://mlflow.org/docs/latest/tracking.html
[3] MLflow Model Registry & Deployment – https://mlflow.org/docs/latest/model-registry.html
```

### Answering query using Neo4J vector DB

- `output/user_questions.md`:

```
What have patients said about their quality of rest during their stay?
```

- `output/final_answer.md`:

```
## Patient Feedback on Quality of Rest During Hospital Stay

Patients consistently report negative experiences with sleep quality during their hospital stay. The primary issues cited are:

1. **Physical discomfort** – uncomfortable beds that hinder restful sleep.
2. **Environmental noise** – constant machine beeping and nighttime interruptions.

### Key Review Excerpts
- **Crystal Johnson** (Smith, Edwards & Obrien): *"The uncomfortable beds made it difficult to get a good night’s sleep during my stay."*
- **Miranda Frey** (Brown‑Golden): *"The uncomfortable beds made it difficult to get a good night’s sleep during my stay."*
- **Jesse Tucker** (Wallace‑Hamilton): *"The constant interruptions during the night, however, affected my ability to rest."*
- **Heather Lewis** (Brown‑Golden): *"The constant beeping of machines in the ward was bothersome and made it difficult to get quality rest."*

### Overall Sentiment
Patients consistently report negative experiences with sleep quality during their hospital stay, citing physical discomfort and environmental noise as the main contributors.

### Sources
[1] HealthcareCypher: Patient Reviews – https://healthcarecypher.com/patient-reviews
```

- `output/user_questions.md`:

```
What have patients said about hospital efficiency? Highlight details from specific reviews.
```

- `output/final_answer.md`:

```
## Hospital Efficiency – Patient Review Highlights

Patients have commented on the efficiency of care at several hospitals. The following excerpts illustrate specific observations:

- **Brown Inc.** – *Jennifer McCall, Stephen Hernandez*: “The medical team was **efficient**.” This direct praise indicates that patients perceived the care process as swift and well‑coordinated.

- **Little‑Spencer** – *Shawn Ellis, Tracy Dalton*: “The well‑organized approach to my treatment contributed to a positive overall experience.” The phrase “well‑organized” implies streamlined processes and timely care, suggesting efficient workflow.

- **Smith, Edwards and Obrien** – *Nancy Nichols, Martin Gilbert*: “The hospital staff went above and beyond to make me comfortable during my stay. They even organized daily activities to keep patients engaged.” While this review focuses on comfort and engagement, it does not explicitly mention efficiency.

- **Smith, Edwards and Obrien** – *Tara Harris, Garrett Gomez*: “My experience at the hospital was positive overall. The medical team was competent, and the facilities were modern and well‑equipped.” This review highlights competence and modern facilities but lacks mention of efficiency.

These insights are drawn from patient reviews retrieved via the HealthcareReview tool.

### Sources

[1] HealthcareReview – Hospital efficiency patient reviews.
```

### Answering question from Neo4J graph DB

- `output/user_questions.md`:

```
Which physician has treated the most patients covered by Cigna?
```

- `output/final_answer.md`:

```
The physician who has treated the most patients covered by Cigna is **Kayla Lawson**, who has seen **10 patients**.

### Sources
[1] HealthcareCypher: Physician Coverage Data – https://healthcarecypher.com/physician-coverage
```

- `output/user_questions.md`:

```
What is the average visit duration for emergency visits in North Carolina?
```

- `output/final_answer.md`:

```
## Average Emergency Visit Duration in North Carolina

The HealthcareCypher sub‑agent reports an average duration of **≈ 21,705 minutes** (about **15 days**) for emergency department visits in North Carolina. This figure likely reflects the total time patients spend in the hospital following an ED visit, rather than the time spent in the ED itself. No specific data source or time period was provided, so the number should be interpreted with caution.

### Sources

[1] HealthcareCypher: Average ED visit duration in North Carolina – 21,705 minutes (no source cited).
```

### Accummulation

- `output/user_questions.md`:

```
Which state had the largest percent increase in Medicaid visits from 2022 to 2023?
```

- `output/final_answer.md`:

```
## Largest Percent Increase in Medicaid Visits (2022‑2023)

The HealthcareCypher sub‑agent reports that **Texas (TX)** had the largest percentage increase in Medicaid visits between 2022 and 2023, with an increase of **≈ 8.82 %**. This figure comes from the HealthcareCypher database’s Medicaid visit statistics by state for the two years.

### Sources

[1] HealthcareCypher: Medicaid visit statistics by state – 2022 and 2023.
```

- `output/user_questions.md`:

```
Which payer provides the most coverage in terms of total billing amount?
```

- `output/final_answer.md`:

```
## Payer with the Highest Total Coverage Amount

The payer that provides the most coverage in terms of total billing amount is **UnitedHealthcare**, with a total coverage amount of **$52,221,646.74**.

### Summary of Coverage Totals
| Payer | Total Coverage Amount |
|-------|------------------------|
| UnitedHealthcare | $52,221,646.74 |
| Cigna | $52,161,684.47 |
| Blue Cross | $51,440,502.96 |
| Medicaid | $50,464,204.50 |
| Aetna | $48,806,964.29 |

The figures above are derived from the healthcare database query on payer coverage totals.

### Source
[1] HealthcareCypher query result.
```

- `output/user_questions.md`:

```
Categorize all patients' reviews into "Positive", "Neutral", and "Negative". Provide totals and percentage of the categories.
```

- `output/final_answer.md`:

```
## Sentiment Categorization of Patient Reviews

The dataset contains four patient reviews, all of which are classified as **Positive**. No reviews were classified as **Neutral** or **Negative**.

| Sentiment | Count | Percentage |
|-----------|-------|------------|
| Positive | 4 | 100 % |
| Neutral | 0 | 0 % |
| Negative | 0 | 0 % |

### Source
[1] HealthcareCypher query result on patient reviews.
```

### Answering question from Neo4J graph AND vector DB

- `output/user_questions.md`:

```
Query the graph database to show me the reviews written by patient 7674
```

- `output/final_answer.md`:

```
## Reviews Written by Patient 7674

The following table summarizes the review(s) authored by patient **7674** in the healthcare graph database:

| Review ID | Hospital | Physician | Review Text |
|-----------|----------|-----------|-------------|
| 644 | Jones, Brown and Murray | Sandra Porter | “The hospital provided exceptional care, but the billing process was confusing and frustrating. Clearer communication about costs would have been appreciated.” |

### Sources
[1] HealthcareCypher query “reviews written by patient 7674”
```

## Email RAG Deep Agent

### Sub-Agent:

![Email RAG StateGraph with Checkpoint](images/EmailRAGStateGraph.png?raw=true "Email RAG StateGraph with Checkpoint")

### Run:

- Takes about 8 minutes to run.

```
$ uv run python -m src.rag_agent.EmailRAG
```

### Main Agent Outputs:

- `output/email_request.md`:

```
Escalation Criteria: There's an immediate risk of electrical, water, or fire damage
Escalation Dollar Criteria: 100000
Escalation Emails: brog@abc.com, bigceo@company.com

Date: Thu, 3 Apr 2025 11:36:10 +0000
From: City of Los Angeles Building and Safety Department <inspections@lacity.gov>
Reply-To: Admin <admin@building-safety.la.com>
To: West Coast Development <admin@west-coast-dev.com>
Cc: Donald Duck <donald@duck.com>, Support <inspections@lacity.gov>
Message-ID: <f967b2d6-1036-11f0-9701-9775a4ad682f@prod.outlook.com>
References: <7d29dafe-1037-11f0-a588-8f6b1b834703@prod.outlook.com> <859cc44e-1037-11f0-b15c-6b9d5cb20c47@prod.outlook.com>
Subject: Project 345678123 - Sunset Luxury Condominiums
Location: Los Angeles, CA
Following an inspection of your site at 456 Sunset Boulevard, we have identified the following building code violations:
Electrical Wiring: Exposed wiring was found in the underground parking garage, posing a safety hazard.
Fire Safety: Insufficient fire extinguishers were available across multiple floors of the structure under construction.
Structural Integrity: The temporary support beams in the eastern wing do not meet the load-bearing standards specified in local building codes.
Required Corrective Actions: Replace or properly secure exposed wiring to meet electrical safety standards.
Install additional fire extinguishers in compliance with fire code requirements. Reinforce or replace temporary support beams
to ensure structural stability. Deadline for Compliance: Violations must be addressed no later than October 31, 2025.
Failure to comply may result in a stop-work order and additional fines.
Contact: For questions or to schedule a re-inspection, please contact the Building and Safety Department at (555) 456-7890 or email inspections@lacity.gov.
```

- `output/final_report.md`:

```
## Key Findings

The email from the City of Los Angeles Building and Safety Department to West Coast Development identifies several building code violations at the Sunset Luxury Condominiums site located at 456 Sunset Boulevard, Los Angeles, CA.

### Violations
[1] **Electrical Wiring**: Exposed wiring was found in the underground parking garage, posing an immediate electrical hazard.
[2] **Fire Safety**: Insufficient fire extinguishers were available across multiple floors of the structure under construction.
[3] **Structural Integrity**: The temporary support beams in the eastern wing do not meet the load‑bearing standards specified in local building codes.

### Corrective Actions
To rectify these violations, the following actions are required:
[1] Replace or properly secure exposed wiring to meet electrical safety standards.
[2] Install additional fire extinguishers in compliance with fire code requirements.
[3] Reinforce or replace temporary support beams to ensure structural stability.

### Deadline
The deadline for compliance is **October 31, 2025**.

### Fines and Penalties
Failure to comply may result in a stop‑work order and additional fines. Given the escalation dollar criteria of $100,000 and the immediate risk of electrical, water, or fire damage, the situation warrants escalation.

### Escalation
- **Escalation Criteria**: Immediate risk of electrical, water, or fire damage.
- **Escalation Dollar Criteria**: $100,000.
- **Escalation Emails**: brog@abc.com, bigceo@company.com.

### Contact
For questions or to schedule a re‑inspection, contact the Building and Safety Department at (555) 456‑7890 or email inspections@lacity.gov.

---

**Prepared by:** Bob, Email Assistant
**Date:** 2026-01-13
```

### Sub-agent Outputs:

- `output/email_extract.md`:

```
{
    "name": "City of Los Angeles Building and Safety Department",
    "phone": null,
    "email": "inspections@lacity.gov",
    "project_id": 345678123,
    "site_location": "456 Sunset Boulevard, Los Angeles, CA",
    "violation_type": "Electrical Wiring, Fire Safety, Structural Integrity",
    "required_changes": "Replace or properly secure exposed wiring to meet electrical safety standards; Install additional fire extinguishers in compliance with fire code requirements; Reinforce or replace temporary support beams to ensure structural stability",
    "max_potential_fine": null,
    "date_of_email": "2025-04-03",
    "compliance_deadline": "2025-10-31"
}
This email warrants an escalation
```

## LangSmith Application trace

- https://smith.langchain.com/

## Local log

- /var/log/ragagent/log

## Diagnostics

- HTTP/3 curl:

```
$ docker run --rm ymuski/curl-http3 curl --http3 --verbose https://<nodeport service>:<nodeport>/health/ready
```

- To build your own HTTP/3 curl: https://curl.se/docs/http3.html

## Neo4J Graph DB

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
