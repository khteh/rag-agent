#!/bin/bash
rm -f /tmp/server.*
openssl req -new -newkey rsa:4096 -x509 -nodes -days 365 -keyout /tmp/server.key -out /tmp/server.crt -subj "/C=SG/ST=Singapore/L=Singapore /O=Kok How Pte. Ltd./OU=PythonFlaskRestAPI/CN=localhost/emailAddress=funcoolgeek@gmail.com" -passin pass:llm-rag-agent
uv run hypercorn --config=/etc/hypercorn.toml --reload src.main:app