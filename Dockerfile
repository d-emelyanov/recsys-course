FROM python:3.9-slim

WORKDIR /home

ADD . /home

RUN cd /home && \
    pip install --upgrade pip && \
    pip install poetry

RUN poetry config virtualenvs.create false && \
    poetry install

EXPOSE 5000

CMD mlflow ui -h 0.0.0.0 -p 5000
