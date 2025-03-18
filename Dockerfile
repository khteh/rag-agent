FROM khteh/ubuntu:latest
MAINTAINER Kok How, Teh <funcoolgeek@gmail.com>
WORKDIR /app
RUN apt update -y --fix-missing
RUN apt upgrade -y
RUN apt install -y python3-tk
ADD src src
ADD Pipfile .
ADD Pipfile.lock .
RUN pipenv install
#RUN curl -sL -o /tmp/gcloud.tgz https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
#RUN tar -xf /tmp/gcloud.tgz
#RUN ./google-cloud-sdk/install.sh -q
#RUN rm -f /tmp/gcloud.tgz
RUN openssl req -new -newkey rsa:4096 -x509 -nodes -days 365 -keyout /tmp/server.key -out /tmp/server.crt -subj "/C=SG/ST=Singapore/L=Singapore /O=Kok How Pte. Ltd./OU=PythonRestAPI/CN=localhost/emailAddress=funcoolgeek@gmail.com" -passin pass:RAGAgent
#RUN ./google-cloud-sdk/bin/gcloud auth activate-service-account --key-file=/etc/service-account.json
EXPOSE 80 443
ENTRYPOINT ["pipenv", "run", "hypercorn", "--config=/etc/hypercorn.toml", "--reload", "src.main:app"]
