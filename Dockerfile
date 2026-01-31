FROM khteh/ubuntu:latest
LABEL org.opencontainers.image.authors="Kok How, Teh <funcoolgeeek@gmail.com>"
WORKDIR /app
ADD data data
ADD src src
ADD pyproject.toml .
ADD uv.lock .
ADD hypercorn.sh .
RUN uv sync
#RUN openssl req -new -newkey rsa:4096 -x509 -nodes -days 365 -keyout /tmp/server.key -out /tmp/server.crt -subj "/C=SG/ST=Singapore/L=Singapore /O=Kok How Pte. Ltd./OU=PythonRestAPI/CN=localhost/emailAddress=funcoolgeek@gmail.com" -passin pass:llm-rag-agent
EXPOSE 443
#ENTRYPOINT ["uv", "run", "hypercorn", "--config=/etc/hypercorn.toml", "--reload", "src.main:app"]
ENTRYPOINT ["./hypercorn.sh"]
