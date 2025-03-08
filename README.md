# LangChain

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
GEMINI_PROJECT_ID="<foo>"
GEMINI_PROJECT_LOCATION="<foo>"
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

## LangSmith Application trace

- https://smith.langchain.com/
