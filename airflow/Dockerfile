FROM apache/airflow:2.11.1-python3.12

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

USER airflow
WORKDIR /opt/recsys

COPY pyproject.toml ./
COPY src ./src

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python -m pip install --no-cache-dir .

ENV PYTHONPATH=/opt/recsys/src