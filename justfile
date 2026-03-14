set dotenv-load := true

default:
    @just --list

install:
    uv sync

etl:
    uv run python -m cart_driven_recsys.etl

train:
    uv run python -m cart_driven_recsys.train

serve:
    uv run python -m cart_driven_recsys.api

airflow-web:
    uv run airflow webserver --port 8080

airflow-scheduler:
    uv run airflow scheduler

airflow-init:
    uv run airflow db migrate

mlflow:
    mlflow server \
        --host 0.0.0.0 \
        --port 5000 \
        --backend-store-uri ./mlruns \
        --default-artifact-root ./artifacts/models

docker-up:
    docker compose up --build -d

docker-down:
    docker compose down

docker-reset:
    docker compose down -v

logs:
    docker compose logs -f

api-logs:
    docker compose logs -f recsys-api

airflow-logs:
    docker compose logs -f airflow-webserver airflow-scheduler

fmt:
    uv run ruff format .

lint:
    uv run ruff check .