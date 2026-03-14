from __future__ import annotations

from datetime import datetime, timedelta

from airflow.decorators import dag, task


@dag(
    dag_id="cart_recsys_train",
    description="train ALS recommender",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["recsys", "training"],
)
def cart_recsys_train_dag():
    @task
    def check_processed() -> None:
        from cart_driven_recsys.config import cfg

        required_paths = [
            cfg.events_clean_dir,
            cfg.purchases_parquet,
            cfg.items_parquet,
            cfg.categories_parquet,
            cfg.stats_json,
            cfg.success_flag,
        ]

        missing_paths = [str(path) for path in required_paths if not path.exists()]

        if missing_paths:
            raise FileNotFoundError(f"missing processed inputs: {missing_paths}")

    @task
    def run_training_task() -> str:
        from cart_driven_recsys.train import run_training

        result = run_training()
        return result.model_path

    @task
    def validate_model_artifact(model_path: str) -> None:
        from cart_driven_recsys.config import cfg

        if not cfg.model_artifact.exists():
            raise FileNotFoundError(f"model artifact not found: {cfg.model_artifact}")

        if str(cfg.model_artifact) != model_path:
            raise ValueError(
                f"unexpected model path: got {model_path}, expected {cfg.model_artifact}"
            )

    processed_ready = check_processed()
    trained_model_path = run_training_task()
    model_validated = validate_model_artifact(trained_model_path)

    processed_ready >> trained_model_path >> model_validated


cart_recsys_train_dag()