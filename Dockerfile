FROM python:latest
MAINTAINER Kok How, Teh <funcoolgeek@gmail.com>
WORKDIR /app
ADD src src
ADD Pipfile .
ADD Pipfile.lock .
RUN pip install pipenv
RUN pipenv install --system --deploy --ignore-pipfile
RUN openssl req -new -newkey rsa:4096 -x509 -nodes -days 365 -keyout /tmp/server.key -out /tmp/server.crt -subj "/C=SG/ST=Singapore/L=Singapore /O=Kok How Pte. Ltd./OU=PythonRestAPI/CN=localhost/emailAddress=funcoolgeek@gmail.com" -passin pass:RAGAgent
EXPOSE 8080 4433
ENTRYPOINT ["hypercorn", "--config=/etc/hypercorn.toml", "--reload", "src.main:app"]
