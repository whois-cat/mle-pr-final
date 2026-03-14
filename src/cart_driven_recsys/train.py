from __future__ import annotations

import logging
from dataclasses import dataclass

import duckdb
import joblib
import mlflow
import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy import sparse

from cart_driven_recsys.config import cfg
from cart_driven_recsys import sql


logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    metrics: dict[str, float | int]
    model_path: str


def _connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


def load_interactions() -> pd.DataFrame:
    connection = _connect()
    try:
        interactions = connection.execute(
            sql.interactions_sql(
                events_clean_dir=cfg.events_clean_dir,
                weight_view=cfg.s.weight_view,
                weight_addtocart=cfg.s.weight_addtocart,
                weight_transaction=cfg.s.weight_transaction,
            )
        ).df()
    finally:
        connection.close()

    interactions["event_time"] = pd.to_datetime(interactions["event_time"], utc=True)

    grouped = (
        interactions.groupby(["user_id", "item_id"], as_index=False)
        .agg(
            weight=("weight", "sum"),
            last_event_time=("event_time", "max"),
        )
    )

    return grouped


def load_raw_addtocart_events() -> pd.DataFrame:
    connection = _connect()
    try:
        events = connection.execute(sql.raw_addtocart_events_sql(cfg.events_clean_dir)).df()
    finally:
        connection.close()

    events["event_time"] = pd.to_datetime(events["event_time"], utc=True)
    return events


def load_popular_items(top_k: int | None = None) -> list[int]:
    connection = _connect()

    try:
        popular_items = connection.execute(
            sql.popular_items_sql(
                events_clean_dir=cfg.events_clean_dir,
                weight_view=cfg.s.weight_view,
                weight_addtocart=cfg.s.weight_addtocart,
                weight_transaction=cfg.s.weight_transaction,
            )
        ).df()
    finally:
        connection.close()

    if top_k is None:
        top_k = cfg.s.popular_items_count

    return popular_items["item_id"].head(top_k).astype(int).tolist()


def split_by_time(
    interactions: pd.DataFrame,
    test_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    if interactions.empty:
        raise ValueError("interactions dataframe is empty")

    cutoff = interactions["last_event_time"].max() - pd.Timedelta(days=test_days)
    train_df = interactions[interactions["last_event_time"] < cutoff].copy()

    test_events = load_raw_addtocart_events()
    test_events = test_events[test_events["event_time"] >= cutoff].copy()

    return train_df, test_events, cutoff


def build_interaction_matrix(
    train_df: pd.DataFrame,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    if train_df.empty:
        raise ValueError("train dataframe is empty")

    user_ids = np.sort(train_df["user_id"].unique())
    item_ids = np.sort(train_df["item_id"].unique())

    user_id_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
    item_id_to_index = {item_id: index for index, item_id in enumerate(item_ids)}

    row_indices = train_df["user_id"].map(user_id_to_index).to_numpy()
    col_indices = train_df["item_id"].map(item_id_to_index).to_numpy()
    values = train_df["weight"].astype(np.float32).to_numpy()

    matrix = sparse.csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(user_ids), len(item_ids)),
        dtype=np.float32,
    )

    return matrix, user_ids, item_ids


def build_sessions_from_events(events: pd.DataFrame, gap_hours: int) -> list[list[int]]:
    if events.empty:
        return []

    ordered_events = events.sort_values(["user_id", "event_time"]).copy()
    session_gap = pd.Timedelta(hours=gap_hours)

    ordered_events["prev_event_time"] = ordered_events.groupby("user_id")["event_time"].shift()
    ordered_events["is_new_session"] = (
        ordered_events["prev_event_time"].isna()
        | ((ordered_events["event_time"] - ordered_events["prev_event_time"]) > session_gap)
    )
    ordered_events["session_id"] = ordered_events.groupby("user_id")["is_new_session"].cumsum()

    sessions_df = (
        ordered_events.groupby(["user_id", "session_id"], sort=False)["item_id"]
        .agg(list)
        .reset_index()
    )

    return [items for items in sessions_df["item_id"] if len(items) >= 2]


def evaluate(
    model: AlternatingLeastSquares,
    item_ids: np.ndarray,
    test_events: pd.DataFrame,
    k: int,
    n_sessions: int,
    gap_hours: int,
) -> dict[str, float | int]:
    sessions = build_sessions_from_events(test_events, gap_hours=gap_hours)

    if not sessions:
        return {f"hit@{k}": 0.0, "mrr": 0.0, "n_sessions": 0}

    rng = np.random.default_rng(42)
    sampled_indices = rng.choice(len(sessions), size=min(n_sessions, len(sessions)), replace=False)
    sampled_sessions = [sessions[index] for index in sampled_indices]

    item_id_to_index = {item_id: index for index, item_id in enumerate(item_ids)}
    
    hit_scores: list[float] = []
    reciprocal_ranks: list[float] = []

    for session in sampled_sessions:
        cart_item_ids = session[:-1]
        target_item_id = session[-1]

        cart_indices = [item_id_to_index[item_id] for item_id in cart_item_ids if item_id in item_id_to_index]

        if not cart_indices or target_item_id not in item_id_to_index:
            hit_scores.append(0.0)
            reciprocal_ranks.append(0.0)
            continue

        user_vector = model.item_factors[cart_indices].mean(axis=0)
        scores = model.item_factors @ user_vector

        for item_id in cart_item_ids:
            if item_id in item_id_to_index:
                scores[item_id_to_index[item_id]] = -np.inf

        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        recommended_item_ids = item_ids[top_indices].tolist()

        if target_item_id in recommended_item_ids:
            rank = recommended_item_ids.index(target_item_id) + 1
            hit_scores.append(1.0)
            reciprocal_ranks.append(1.0 / rank)
        else:
            hit_scores.append(0.0)
            reciprocal_ranks.append(0.0)

    return {
        f"hit@{k}": float(np.mean(hit_scores)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "n_sessions": len(sampled_sessions),
    }


def train_als_model(interaction_matrix: sparse.csr_matrix) -> AlternatingLeastSquares:
    model = AlternatingLeastSquares(
        factors=cfg.s.als_factors,
        iterations=cfg.s.als_iterations,
        regularization=cfg.s.als_regularization,
        alpha=cfg.s.als_alpha,
        use_gpu=False,
        calculate_training_loss=True,
        random_state=42,
    )
    model.fit(interaction_matrix)
    return model


def save_model_artifact(
    model: AlternatingLeastSquares,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    popular_items: list[int],
    cutoff: pd.Timestamp,
    interaction_matrix: sparse.csr_matrix,
) -> None:
    artifact = {
        "model": model,
        "item_factors": model.item_factors,
        "user_factors": model.user_factors,
        "user_ids": user_ids,
        "item_ids": item_ids,
        "popular_items": popular_items,
        "train_meta": {
            "cutoff_date": str(cutoff.date()),
            "n_users": int(interaction_matrix.shape[0]),
            "n_items": int(interaction_matrix.shape[1]),
        },
    }

    cfg.model_artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, cfg.model_artifact)


def log_to_mlflow(metrics: dict[str, float | int]) -> None:
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.s.mlflow_experiment)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "weight_view": cfg.s.weight_view,
                "weight_addtocart": cfg.s.weight_addtocart,
                "weight_transaction": cfg.s.weight_transaction,
                "als_factors": cfg.s.als_factors,
                "als_iterations": cfg.s.als_iterations,
                "als_regularization": cfg.s.als_regularization,
                "als_alpha": cfg.s.als_alpha,
                "eval_test_days": cfg.s.eval_test_days,
                "eval_session_gap_hours": cfg.s.eval_session_gap_hours,
                "eval_k": cfg.s.eval_k,
                "eval_n_sessions": cfg.s.eval_n_sessions,
            }
        )

        mlflow.log_metrics(
            {
                metric_name: float(metric_value)
                for metric_name, metric_value in metrics.items()
                if isinstance(metric_value, (int, float))
            }
        )

        mlflow.log_artifact(str(cfg.model_artifact))


def run_training() -> TrainingResult:
    logger.info("train: load interactions")
    interactions = load_interactions()

    logger.info("train: split by time")
    train_df, test_events, cutoff = split_by_time(
        interactions=interactions,
        test_days=cfg.s.eval_test_days,
    )

    logger.info("train: build interaction matrix")
    interaction_matrix, user_ids, item_ids = build_interaction_matrix(train_df)

    logger.info("train: load popular items")
    popular_items = load_popular_items()

    logger.info("train: fit als")
    model = train_als_model(interaction_matrix)

    logger.info("train: evaluate")
    metrics = evaluate(
        model=model,
        item_ids=item_ids,
        test_events=test_events,
        k=cfg.s.eval_k,
        n_sessions=cfg.s.eval_n_sessions,
        gap_hours=cfg.s.eval_session_gap_hours,
    )

    logger.info("train: save model artifact")
    save_model_artifact(
        model=model,
        user_ids=user_ids,
        item_ids=item_ids,
        popular_items=popular_items,
        cutoff=cutoff,
        interaction_matrix=interaction_matrix,
    )

    logger.info("train: log to mlflow")
    log_to_mlflow(metrics)

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