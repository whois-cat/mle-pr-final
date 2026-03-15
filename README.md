# Cart-driven RecSys

This project takes raw clickstream csv files, builds parquet datasets with duckdb, trains an implicit-feedback als model, adds a co-visitation signal, serves top-k recommendations through fastapi, and exposes metrics for mlflow, prometheus, and grafana.

## What this project does

- cleans raw event data and writes partitioned parquet
- builds derived datasets for purchases, items, categories, and dataset stats
- trains a hybrid recommender for add-to-cart prediction
- stores a single model artifact with model + metadata + fallback data
- serves recommendations through an http api
- ships a local stack with airflow, mlflow, prometheus, and grafana

## Stack

- python 3.12
- duckdb for sql-based etl
- pandas / numpy / scipy
- implicit als for collaborative filtering
- fastapi + uvicorn for serving
- mlflow for experiment tracking
- airflow for training orchestration
- prometheus + grafana for monitoring
- docker compose for local infrastructure

## Project layout

```text
airflow/                    airflow image and dags
monitoring/                 prometheus config + grafana provisioning
notebooks/                  exploration notebooks
src/cart_driven_recsys/
  api.py                    fastapi service
  config.py                 paths and runtime settings
  covisit.py                co-visitation index builder
  etl.py                    duckdb-based preprocessing
  metrics.py                prometheus metrics
  recommenders.py           retrieval, fusion, evaluation
  sql.py                    sql templates for etl and training reads
  train.py                  training, evaluation, mlflow logging
tests/                      api tests
Dockerfile                  api image
pyproject.toml              dependencies and package config
justfile                    helper commands
```

## Ranking logic

1. raw events are converted to weighted implicit feedback
   - `view = 1.0`
   - `addtocart = 5.0`
   - `transaction = 10.0`
2. the training split is built from historical interactions
3. an als model learns item embeddings from the user-item matrix
4. a co-visitation index is built from add-to-cart sessions
5. final recommendations are fused from als + co-visitation with reciprocal rank fusion
6. popular items are used as a fallback if the hybrid result is too short

## Expected raw data

put the source files into `data/raw/` with these names:

```text
events.csv
item_properties_part1.csv
item_properties_part2.csv
category_tree.csv
```

## generated artifacts

after preprocessing and training, the project writes:

```text
data/processed/events_clean/         # partitioned parquet by event_date
data/processed/purchases.parquet
data/processed/items.parquet
data/processed/categories.parquet
data/processed/stats.json
data/processed/_SUCCESS
artifacts/models/als_v1/model.pkl
```

## quick start with docker compose

```bash
cp .env.template .env
mkdir -p data/raw data/processed artifacts/models

docker compose build
docker compose up -d
```

run preprocessing:

```bash
docker compose exec airflow-api-server python -m cart_driven_recsys.etl
```

trigger training:

```bash
docker compose exec airflow-api-server airflow dags trigger cart_recsys_train
```

open the services:

- airflow: `http://localhost:8080`
- mlflow: `http://localhost:5000`
- api docs: `http://localhost:8000/docs`
- prometheus: `http://localhost:9090`
- grafana: `http://localhost:3000`

## local dev

```bash
uv sync --dev
cp .env.template .env
python -m cart_driven_recsys.etl
python -m cart_driven_recsys.train
python -m uvicorn cart_driven_recsys.api:app --host 0.0.0.0 --port 8000
```

## API

### health

```http
GET /health
```

response:

```json
{
  "status": "ok"
}
```

### metadata

```http
GET /metadata
```

returns model metadata such as artifact_path, number of items and users, cutoff date, als factors, and hybrid weights.

### metrics

```http
GET /metrics
```

prometheus-compatible metrics endpoint

### cart recommendations

```http
POST /recommendations/cart
content-type: application/json
```

request body:

```json
{
  "item_ids": [101, 202, 303],
  "k": 10
}
```

response body:

```json
{
  "item_ids": [404, 505, 606],
  "unknown_item_ids": []
}
```

notes:

- duplicate cart items are removed before ranking
- unknown item ids are reported separately
- `k` must be between `1` and `100`

## Training and evaluation

training saves one artifact with:

- the als model
- item ids used by the model
- popularity fallback list
- co-visitation index
- hybrid weights
- training metadata

current evaluation is session-based and uses the last add-to-cart item in a sampled session as the target.

## Reported metrics:

- `hit@k`
- `ndcg@k`
- `coverage@k`
- `popularity_bias_top_1pct`
- `n_sessions`

## Monitoring

prometheus scrapes the api `/metrics` endpoint.

grafana is provisioned automatically and includes an overview dashboard with panels for:

- request rate
- recommendation throughput
- recommendation error rate
- p99 recommendation latency
- http latency by quantile

## Tests

API tests cover:

- `/health`
- `/metadata`
- `/metrics`
- recommendation response shape
- duplicate removal
- invalid `k`
- error handling for artifact loading and ranking failures

run tests:

```bash
pytest -vv -ra
```
or

```bash
just test
```
## Helper commands

```bash
just up
just down
just build
just rebuild
just logs
just api
just test
just health
```
