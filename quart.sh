#!/bin/bash
rm -f /tmp/server.*
openssl req -new -newkey rsa:4096 -x509 -nodes -days 365 -keyout /tmp/server.key -out /tmp/server.crt -subj "/C=SG/ST=Singapore/L=Singapore /O=Kok How Pte. Ltd./OU=PythonFlaskRestAPI/CN=localhost/emailAddress=funcoolgeek@gmail.com" -passin pass:RAGAgent
#pipenv run hypercorn --config=/etc/pythonrestapi.toml --quic-bind 0.0.0.0:443 --certfile server.crt --keyfile server.key --bind 0.0.0.0:80 --reload  src.main:app
pipenv run hypercorn --reload src.main:app
