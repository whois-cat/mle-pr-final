from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARAMS_PATH = ROOT / "params.yaml"


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="RECSYS_",
    )

    data_raw_dir: Path = ROOT / "data" / "raw"
    data_processed_dir: Path = ROOT / "data" / "processed"
    artifacts_dir: Path = ROOT / "artifacts" / "models"
    params_file: Path = DEFAULT_PARAMS_PATH

    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment: str = "cart-driven-recsys"
    api_host: str = "0.0.0.0"
    api_port: int = 8000


class DataParams(BaseModel):
    events_file: str = "events.csv"
    item_props_files: list[str] = Field(
        default_factory=lambda: ["item_properties_part1.csv", "item_properties_part2.csv"]
    )
    category_tree_file: str = "category_tree.csv"


class ProcessedParams(BaseModel):
    events_clean_dirname: str = "events_clean"
    purchases_filename: str = "purchases.parquet"
    items_filename: str = "items.parquet"
    categories_filename: str = "categories.parquet"
    stats_filename: str = "stats.json"
    success_filename: str = "_SUCCESS"
    duckdb_filename: str = "recsys.duckdb"


class ModelParams(BaseModel):
    name: str = "als_v1"
    type: str = "hybrid_als_covisit"
    artifact_filename: str = "model.pkl"


class WeightsParams(BaseModel):
    view: float = 1.0
    addtocart: float = 5.0
    transaction: float = 10.0


class AlsParams(BaseModel):
    factors: int = 128
    iterations: int = 30
    regularization: float = 0.01
    alpha: float = 40.0


class CovisitParams(BaseModel):
    top_neighbors: int = 50


class HybridParams(BaseModel):
    als_weight: float = 0.7
    covisit_weight: float = 0.3
    rrf_constant: int = 60


class EvalParams(BaseModel):
    test_days: int = 30
    session_gap_hours: int = 4
    k: int = 10
    n_sessions: int = 5000


class PopularParams(BaseModel):
    items_count: int = 500


class ParamsFile(BaseModel):
    data: DataParams = Field(default_factory=DataParams)
    processed: ProcessedParams = Field(default_factory=ProcessedParams)
    model: ModelParams = Field(default_factory=ModelParams)
    weights: WeightsParams = Field(default_factory=WeightsParams)
    als: AlsParams = Field(default_factory=AlsParams)
    covisit: CovisitParams = Field(default_factory=CovisitParams)
    hybrid: HybridParams = Field(default_factory=HybridParams)
    eval: EvalParams = Field(default_factory=EvalParams)
    popular: PopularParams = Field(default_factory=PopularParams)


def _load_params(params_path: Path) -> ParamsFile:
    if not params_path.exists():
        raise FileNotFoundError(f"params file not found: {params_path}")

    loaded_params = yaml.safe_load(params_path.read_text(encoding="utf-8")) or {}
    return ParamsFile.model_validate(loaded_params)


class Cfg:
    def __init__(self, env_settings: EnvSettings, params_model: ParamsFile, params_path: Path):
        if len(params_model.data.item_props_files) != 2:
            raise ValueError("data.item_props_files must contain exactly 2 filenames")

        self.root_dir = ROOT
        self.params_path = params_path
        self.env = env_settings
        self.params_model = params_model
        self.params = params_model.model_dump()

        self.data_raw_dir = env_settings.data_raw_dir
        self.data_processed_dir = env_settings.data_processed_dir
        self.artifacts_dir = env_settings.artifacts_dir

        self.events_csv = self.data_raw_dir / params_model.data.events_file
        self.item_props_csvs = tuple(
            self.data_raw_dir / file_name
            for file_name in params_model.data.item_props_files
        )
        self.category_tree_csv = self.data_raw_dir / params_model.data.category_tree_file

        self.events_clean_dir = self.data_processed_dir / params_model.processed.events_clean_dirname
        self.purchases_parquet = self.data_processed_dir / params_model.processed.purchases_filename
        self.items_parquet = self.data_processed_dir / params_model.processed.items_filename
        self.categories_parquet = self.data_processed_dir / params_model.processed.categories_filename
        self.stats_json = self.data_processed_dir / params_model.processed.stats_filename
        self.success_flag = self.data_processed_dir / params_model.processed.success_filename
        self.duckdb_path = self.data_processed_dir / params_model.processed.duckdb_filename

        self.model_dir = self.artifacts_dir / params_model.model.name
        self.model_artifact = self.model_dir / params_model.model.artifact_filename

        self.mlflow_tracking_uri = env_settings.mlflow_tracking_uri
        self.mlflow_experiment = env_settings.mlflow_experiment
        self.api_host = env_settings.api_host
        self.api_port = env_settings.api_port


@lru_cache(maxsize=1)
def _build_cfg() -> Cfg:
    env_settings = EnvSettings()
    params_path = env_settings.params_file

    if not params_path.is_absolute():
        params_path = ROOT / params_path

    return Cfg(
        env_settings=env_settings,
        params_model=_load_params(params_path),
        params_path=params_path,
    )


cfg = _build_cfg()
