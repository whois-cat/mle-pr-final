FROM apache/airflow:2.11.1-python3.12

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/recsys

COPY pyproject.toml ./
COPY src ./src

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python -m pip install --no-cache-dir .

RUN chown -R airflow:root /opt/recsys

USER airflow

ENV PYTHONPATH=/opt/recsys/src