set shell := ["bash", "-cu"]

compose := "docker compose"

up:
    {{compose}} up -d

down:
    {{compose}} down

build:
    {{compose}} build

rebuild:
    {{compose}} down
    {{compose}} build --no-cache
    {{compose}} up -d

logs:
    {{compose}} logs -f

api:
    {{compose}} up -d recsys-api

train:
    {{compose}} exec airflow-webserver airflow dags trigger cart_recsys_train

test:
    pytest -vv -ra --durations=10 --color=yes

health:
    {{compose}} ps
    echo "airflow: http://localhost:8080"
    echo "mlflow: http://localhost:5000"
    echo "api: http://localhost:8000/docs"
    echo "grafana: http://localhost:3000"
    echo "prom: http://localhost:9090"