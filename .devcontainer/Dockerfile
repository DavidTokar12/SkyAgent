FROM mcr.microsoft.com/devcontainers/python:3.12
ARG DEBIAN_FRONTEND=noninteractive
ARG USER=vscode

COPY system-requirements.txt /tmp/system-requirements.txt
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    xargs -a /tmp/system-requirements.txt apt-get install -y --no-install-recommends && \
    rm -f /tmp/system-requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -