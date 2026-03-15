from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import duckdb
import joblib
import mlflow
import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy import sparse

from cart_driven_recsys import sql
from cart_driven_recsys.config import cfg
from cart_driven_recsys.covisit import build_covisit_index
from cart_driven_recsys.recommenders import (
    build_item_popularity_ranking,
    evaluate_recommender,
    recommend_with_hybrid,
    sample_sessions,
)


logger = logging.getLogger(__name__)

weights = cfg.params["weights"]
als_params = cfg.params["als"]
covisit_params = cfg.params["covisit"]
hybrid_params = cfg.params["hybrid"]
eval_params = cfg.params["eval"]
model_params = cfg.params["model"]
popular_params = cfg.params["popular"]


@dataclass
class TrainingResult:
    metrics: dict[str, float | int]
    model_path: str


def _query_dataframe(sql_query: str) -> pd.DataFrame:
    database_connection = duckdb.connect()
    try:
        return database_connection.execute(sql_query).df()
    finally:
        database_connection.close()


def load_weighted_events() -> pd.DataFrame:
    weighted_events = _query_dataframe(
        sql.interactions_sql(
            events_clean_dir=cfg.events_clean_dir,
            weight_view=weights["view"],
            weight_addtocart=weights["addtocart"],
            weight_transaction=weights["transaction"],
        )
    )
    weighted_events["event_time"] = pd.to_datetime(weighted_events["event_time"], utc=True)
    return weighted_events


def load_raw_addtocart_events() -> pd.DataFrame:
    raw_addtocart_events = _query_dataframe(
        sql.raw_addtocart_events_sql(cfg.events_clean_dir)
    )
    raw_addtocart_events["event_time"] = pd.to_datetime(
        raw_addtocart_events["event_time"],
        utc=True,
    )
    return raw_addtocart_events


def aggregate_interactions(weighted_events: pd.DataFrame) -> pd.DataFrame:
    if weighted_events.empty:
        return pd.DataFrame(columns=["user_id", "item_id", "weight", "last_event_time"])

    return (
        weighted_events.groupby(["user_id", "item_id"], as_index=False)
        .agg(
            weight=("weight", "sum"),
            last_event_time=("event_time", "max"),
        )
    )


def split_by_time(
    weighted_events: pd.DataFrame,
    test_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    if weighted_events.empty:
        raise ValueError("weighted events dataframe is empty")

    cutoff_timestamp = weighted_events["event_time"].max() - pd.Timedelta(days=test_days)

    train_weighted_events = weighted_events[weighted_events["event_time"] < cutoff_timestamp].copy()
    if train_weighted_events.empty:
        raise ValueError("train weighted events dataframe is empty after time split")

    train_interactions = aggregate_interactions(train_weighted_events)

    test_events = load_raw_addtocart_events()
    test_events = test_events[test_events["event_time"] >= cutoff_timestamp].copy()

    return train_interactions, test_events, cutoff_timestamp


def build_interaction_matrix(
    train_interactions: pd.DataFrame,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    if train_interactions.empty:
        raise ValueError("train interactions dataframe is empty")

    unique_user_ids = np.sort(train_interactions["user_id"].unique())
    item_ids = np.sort(train_interactions["item_id"].unique())

    user_id_to_index = {
        int(user_id): user_index
        for user_index, user_id in enumerate(unique_user_ids)
    }
    item_id_to_index = {
        int(item_id): item_index
        for item_index, item_id in enumerate(item_ids)
    }

    row_indices = train_interactions["user_id"].map(user_id_to_index).to_numpy()
    column_indices = train_interactions["item_id"].map(item_id_to_index).to_numpy()
    interaction_weights = train_interactions["weight"].astype(np.float32).to_numpy()

    interaction_matrix = sparse.csr_matrix(
        (interaction_weights, (row_indices, column_indices)),
        shape=(len(unique_user_ids), len(item_ids)),
        dtype=np.float32,
    )

    return interaction_matrix, item_ids


def train_als_model(interaction_matrix: sparse.csr_matrix) -> AlternatingLeastSquares:
    model = AlternatingLeastSquares(
        factors=als_params["factors"],
        iterations=als_params["iterations"],
        regularization=als_params["regularization"],
        alpha=als_params["alpha"],
        use_gpu=False,
        calculate_training_loss=True,
        random_state=42,
    )
    model.fit(interaction_matrix)
    return model


def save_model_artifact(
    model: AlternatingLeastSquares,
    item_ids: np.ndarray,
    popular_items: list[int],
    covisit_index: dict[int, list[tuple[int, float]]],
    cutoff_timestamp: pd.Timestamp,
    interaction_matrix: sparse.csr_matrix,
) -> None:
    artifact_payload = {
        "model_type": model_params["type"],
        "als_model": model,
        "item_ids": item_ids,
        "popular_items": popular_items,
        "covisit_index": covisit_index,
        "hybrid_params": {
            "als_weight": hybrid_params["als_weight"],
            "covisit_weight": hybrid_params["covisit_weight"],
            "rrf_constant": hybrid_params["rrf_constant"],
        },
        "train_meta": {
            "cutoff_date": str(cutoff_timestamp.date()),
            "n_users": int(interaction_matrix.shape[0]),
            "n_items": int(interaction_matrix.shape[1]),
            "als_factors": als_params["factors"],
            "covisit_top_neighbors": covisit_params["top_neighbors"],
        },
    }

    cfg.model_artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact_payload, cfg.model_artifact)


def sanitize_metric_names_for_mlflow(metrics: dict[str, float | int]) -> dict[str, float]:
    sanitized_metrics: dict[str, float] = {}

    for metric_name, metric_value in metrics.items():
        if not isinstance(metric_value, (int, float)):
            continue

        safe_metric_name = metric_name.replace("@", "_at_")
        safe_metric_name = re.sub(r"[^a-zA-Z0-9_\-./ ]", "_", safe_metric_name)
        sanitized_metrics[safe_metric_name] = float(metric_value)

    return sanitized_metrics


def _build_mlflow_params() -> dict[str, float | int | str]:
    return {
        "model_name": model_params["name"],
        "model_type": model_params["type"],
        "weight_view": weights["view"],
        "weight_addtocart": weights["addtocart"],
        "weight_transaction": weights["transaction"],
        "als_factors": als_params["factors"],
        "als_iterations": als_params["iterations"],
        "als_regularization": als_params["regularization"],
        "als_alpha": als_params["alpha"],
        "covisit_top_neighbors": covisit_params["top_neighbors"],
        "hybrid_als_weight": hybrid_params["als_weight"],
        "hybrid_covisit_weight": hybrid_params["covisit_weight"],
        "hybrid_rrf_constant": hybrid_params["rrf_constant"],
        "eval_test_days": eval_params["test_days"],
        "eval_session_gap_hours": eval_params["session_gap_hours"],
        "eval_k": eval_params["k"],
        "eval_n_sessions": eval_params["n_sessions"],
        "popular_items_count": popular_params["items_count"],
    }


def log_to_mlflow(metrics: dict[str, float | int]) -> None:
    if not cfg.mlflow_tracking_uri:
        logger.info("train: skip mlflow logging because tracking uri is empty")
        return

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)

    with mlflow.start_run():
        mlflow.log_params(_build_mlflow_params())
        mlflow.log_metrics(sanitize_metric_names_for_mlflow(metrics))
        mlflow.log_artifact(str(cfg.model_artifact))


def run_training() -> TrainingResult:
    logger.info("train: load weighted events")
    weighted_events = load_weighted_events()

    logger.info("train: split by time")
    train_interactions, test_events, cutoff_timestamp = split_by_time(
        weighted_events=weighted_events,
        test_days=eval_params["test_days"],
    )

    logger.info("train: build interaction matrix")
    interaction_matrix, item_ids = build_interaction_matrix(train_interactions)

    logger.info("train: build popularity ranking")
    full_popular_items = build_item_popularity_ranking(train_interactions)
    fallback_popular_items = full_popular_items[: popular_params["items_count"]]

    logger.info("train: load raw add-to-cart events")
    raw_addtocart_events = load_raw_addtocart_events()
    train_addtocart_events = raw_addtocart_events[
        raw_addtocart_events["event_time"] < cutoff_timestamp
    ].copy()

    logger.info("train: build co-visitation index")
    covisit_index = build_covisit_index(
        train_events=train_addtocart_events,
        gap_hours=eval_params["session_gap_hours"],
        top_neighbors_per_item=covisit_params["top_neighbors"],
    )

    logger.info("train: sample sessions")
    sampled_sessions = sample_sessions(
        test_events=test_events,
        gap_hours=eval_params["session_gap_hours"],
        n_sessions=eval_params["n_sessions"],
        random_seed=42,
    )

    logger.info("train: fit als")
    model = train_als_model(interaction_matrix)

    logger.info("train: evaluate")
    item_id_to_index = {
        int(item_id): item_index
        for item_index, item_id in enumerate(item_ids)
    }

    metrics = evaluate_recommender(
        recommend_function=lambda cart_item_ids, k: recommend_with_hybrid(
            model=model,
            item_ids=item_ids,
            item_id_to_index=item_id_to_index,
            covisit_index=covisit_index,
            popular_items=fallback_popular_items,
            cart_item_ids=cart_item_ids,
            k=k,
            als_weight=hybrid_params["als_weight"],
            covisit_weight=hybrid_params["covisit_weight"],
            rrf_constant=hybrid_params["rrf_constant"],
        ),
        sampled_sessions=sampled_sessions,
        item_catalog_ids=item_ids,
        popular_items=full_popular_items,
        k=eval_params["k"],
    )

    logger.info("train: save model artifact")
    save_model_artifact(
        model=model,
        item_ids=item_ids,
        popular_items=fallback_popular_items,
        covisit_index=covisit_index,
        cutoff_timestamp=cutoff_timestamp,
        interaction_matrix=interaction_matrix,
    )

    try:
        logger.info("train: log to mlflow")
        log_to_mlflow(metrics)
    except Exception:
        logger.exception("train: mlflow logging failed")

    logger.info("train: done")
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}={metric_value}")

    return TrainingResult(
        metrics=metrics,
        model_path=str(cfg.model_artifact),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_training()


if __name__ == "__main__":
    main()
