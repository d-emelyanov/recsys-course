FROM python:3.9-slim

WORKDIR /home

ADD . /home

RUN apt-get update && apt-get install -y git build-essential

RUN cd /home && \
    pip install --upgrade pip && \
    pip install poetry

RUN poetry config virtualenvs.create false && \
    poetry install
