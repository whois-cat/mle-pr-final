from __future__ import annotations

from datetime import datetime, timedelta
from airflow.decorators import dag, task
from cart_driven_recsys.config import cfg


@dag(
    dag_id="cart_recsys_etl",
    description="etl",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["recsys", "etl"],
)
def cart_recsys_etl_dag():
    @task
    def check_raw() -> None:
        required_files = [
            cfg.events_csv,
            cfg.item_props_csvs[0],
            cfg.item_props_csvs[1],
            cfg.category_tree_csv,
        ]

        missing_files = [
            str(file_path)
            for file_path in required_files
            if (not file_path.exists()) or file_path.stat().st_size == 0
        ]

        if missing_files:
            raise FileNotFoundError(f"missing or empty raw files: {missing_files}")

    @task
    def run_etl() -> None:
        from cart_driven_recsys.etl import run_all

        run_all()

    @task
    def validate_outputs() -> None:
        required_outputs = [
            cfg.events_clean_dir,
            cfg.purchases_parquet,
            cfg.items_parquet,
            cfg.categories_parquet,
            cfg.stats_json,
            cfg.success_flag,
        ]

        missing_outputs = [str(path) for path in required_outputs if not path.exists()]

        if missing_outputs:
            raise FileNotFoundError(f"missing outputs: {missing_outputs}")

    check_raw() >> run_etl() >> validate_outputs()


cart_recsys_etl_dag()