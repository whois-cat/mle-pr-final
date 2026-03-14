FROM apache/airflow:3.1.8

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

USER airflow
WORKDIR /opt/airflow

COPY pyproject.toml uv.lock ./
COPY src ./src
COPY airflow ./airflow

RUN pip install --no-cache-dir uv && \
    uv sync --frozen --extra airflow

ENV PYTHONPATH=/opt/airflow/src