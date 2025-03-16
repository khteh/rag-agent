#!/bin/bash
#langgraph build -t khteh/ragagent
docker build -t khteh/ragagent .
docker push khteh/ragagent:latest