#!/bin/bash
langgraph build -t khteh/rag_agent
docker push khteh/rag_agent:latest