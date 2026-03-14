from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_raw_dir: Path = ROOT / "data" / "raw"
    data_processed_dir: Path = ROOT / "data" / "processed"
    artifacts_dir: Path = ROOT / "artifacts" / "models"

    events_file: str = "events.csv"
    item_props_files: list[str] = ["item_properties_part1.csv", "item_properties_part2.csv"]
    category_tree_file: str = "category_tree.csv"

    events_clean_dirname: str = "events_clean"
    purchases_filename: str = "purchases.parquet"
    items_filename: str = "items.parquet"
    categories_filename: str = "categories.parquet"
    stats_filename: str = "stats.json"
    success_filename: str = "_SUCCESS"

    weight_view: float = 1.0
    weight_addtocart: float = 5.0
    weight_transaction: float = 10.0

    min_user_weight: float = 3.0
    min_item_weight: float = 3.0

    duckdb_path: Path = ROOT / "data" / "processed" / "recsys.duckdb"

    als_factors: int = 128
    als_iterations: int = 30
    als_regularization: float = 0.01
    als_alpha: float = 40.0

    model_name: str = "als_v1"
    model_artifact_filename: str = "model.pkl"

    eval_test_days: int = 30
    eval_session_gap_hours: int = 4
    eval_k: int = 10
    eval_n_sessions: int = 5000

    popular_items_count: int = 500

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 2

    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment: str = "cart-driven-recsys"


class Cfg:
    def __init__(self, settings: Settings):
        self.s = settings

        self.events_csv = settings.data_raw_dir / settings.events_file
        self.item_props_csvs = tuple(settings.data_raw_dir / name for name in settings.item_props_files)
        self.category_tree_csv = settings.data_raw_dir / settings.category_tree_file

        self.events_clean_dir = settings.data_processed_dir / settings.events_clean_dirname
        self.purchases_parquet = settings.data_processed_dir / settings.purchases_filename
        self.items_parquet = settings.data_processed_dir / settings.items_filename
        self.categories_parquet = settings.data_processed_dir / settings.categories_filename
        self.stats_json = settings.data_processed_dir / settings.stats_filename
        self.success_flag = settings.data_processed_dir / settings.success_filename

        self.model_dir = settings.artifacts_dir / settings.model_name
        self.model_artifact = self.model_dir / settings.model_artifact_filename

        self.mlflow_tracking_uri = settings.mlflow_tracking_uri

        if len(self.item_props_csvs) != 2:
            raise ValueError("item_props_files must contain exactly 2 filenames")


@lru_cache(maxsize=1)
def _build_cfg() -> Cfg:
    return Cfg(Settings())


cfg = _build_cfg()