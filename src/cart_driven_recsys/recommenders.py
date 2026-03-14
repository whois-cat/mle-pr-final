from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares


def build_item_popularity_ranking(train_interactions: pd.DataFrame) -> list[int]:
    if train_interactions.empty:
        return []

    popularity = (
        train_interactions.groupby("item_id", as_index=False)["weight"]
        .sum()
        .sort_values(["weight", "item_id"], ascending=[False, True])
    )

    return popularity["item_id"].astype(int).tolist()


def build_sessions_from_events(events: pd.DataFrame, gap_hours: int) -> list[list[int]]:
    if events.empty:
        return []

    events = events.sort_values(["user_id", "event_time"]).copy()
    session_gap = pd.Timedelta(hours=gap_hours)

    events["previous_event_time"] = events.groupby("user_id")["event_time"].shift()
    events["is_new_session"] = (
        events["previous_event_time"].isna()
        | ((events["event_time"] - events["previous_event_time"]) > session_gap)
    )
    events["session_id"] = events.groupby("user_id")["is_new_session"].cumsum()

    sessions = (
        events.groupby(["user_id", "session_id"], sort=False)["item_id"]
        .agg(list)
        .reset_index()
    )

    return [item_ids for item_ids in sessions["item_id"] if len(item_ids) >= 2]


def sample_sessions(
    test_events: pd.DataFrame,
    gap_hours: int,
    n_sessions: int,
    random_seed: int = 42,
) -> list[list[int]]:
    sessions = build_sessions_from_events(test_events, gap_hours=gap_hours)

    if not sessions:
        return []

    random_generator = np.random.default_rng(random_seed)
    sampled_indices = random_generator.choice(
        len(sessions),
        size=min(n_sessions, len(sessions)),
        replace=False,
    )

    return [sessions[int(index)] for index in sampled_indices]


def recommend_with_popularity(
    popular_items: list[int],
    cart_item_ids: list[int],
    k: int,
) -> list[int]:
    excluded = set(int(item_id) for item_id in cart_item_ids)

    recommendations: list[int] = []
    for item_id in popular_items:
        if item_id in excluded:
            continue

        recommendations.append(int(item_id))

        if len(recommendations) == k:
            break

    return recommendations


def recommend_with_als(
    model: AlternatingLeastSquares,
    item_ids: np.ndarray,
    item_id_to_index: dict[int, int],
    cart_item_ids: list[int],
    k: int,
) -> list[int]:
    cart_indices = sorted(
        {
            item_id_to_index[item_id]
            for item_id in cart_item_ids
            if item_id in item_id_to_index
        }
    )

    if not cart_indices:
        return []

    recommendation_count = min(k, len(item_ids) - len(cart_indices))
    if recommendation_count <= 0:
        return []

    cart_vector = model.item_factors[cart_indices].mean(axis=0)
    scores = model.item_factors @ cart_vector
    scores = scores.astype(np.float32, copy=False)
    scores[cart_indices] = -np.inf

    top_indices = np.argpartition(scores, -recommendation_count)[-recommendation_count:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    return item_ids[top_indices].astype(int).tolist()


def recommend_with_covisit(
    covisit_index: dict[int, list[tuple[int, float]]],
    cart_item_ids: list[int],
    k: int,
) -> list[int]:
    unique_cart_item_ids = list(dict.fromkeys(int(item_id) for item_id in cart_item_ids))
    excluded = set(unique_cart_item_ids)

    scores: dict[int, float] = {}

    for reverse_position, cart_item_id in enumerate(reversed(unique_cart_item_ids), start=1):
        recency_weight = 1.0 / reverse_position

        for neighbor_item_id, neighbor_score in covisit_index.get(cart_item_id, []):
            if neighbor_item_id in excluded:
                continue

            scores[neighbor_item_id] = scores.get(neighbor_item_id, 0.0)
            scores[neighbor_item_id] += recency_weight * neighbor_score

    ranked_items = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [item_id for item_id, _ in ranked_items[:k]]


def fuse_ranked_lists(
    ranked_lists: list[tuple[list[int], float]],
    k: int,
    rrf_constant: int,
    exclude_item_ids: list[int],
) -> list[int]:
    excluded = set(int(item_id) for item_id in exclude_item_ids)
    fused_scores: dict[int, float] = {}

    for ranked_item_ids, weight in ranked_lists:
        for rank, item_id in enumerate(ranked_item_ids, start=1):
            if item_id in excluded:
                continue

            fused_scores[item_id] = fused_scores.get(item_id, 0.0)
            fused_scores[item_id] += weight / (rrf_constant + rank)

    ranked_items = sorted(fused_scores.items(), key=lambda item: (-item[1], item[0]))
    return [item_id for item_id, _ in ranked_items[:k]]


def recommend_with_hybrid(
    model: AlternatingLeastSquares,
    item_ids: np.ndarray,
    item_id_to_index: dict[int, int],
    covisit_index: dict[int, list[tuple[int, float]]],
    popular_items: list[int],
    cart_item_ids: list[int],
    k: int,
    als_weight: float,
    covisit_weight: float,
    rrf_constant: int,
) -> list[int]:
    als_candidates = recommend_with_als(
        model=model,
        item_ids=item_ids,
        item_id_to_index=item_id_to_index,
        cart_item_ids=cart_item_ids,
        k=max(k * 3, 50),
    )

    covisit_candidates = recommend_with_covisit(
        covisit_index=covisit_index,
        cart_item_ids=cart_item_ids,
        k=max(k * 3, 50),
    )

    recommendations = fuse_ranked_lists(
        ranked_lists=[
            (als_candidates, als_weight),
            (covisit_candidates, covisit_weight),
        ],
        k=k,
        rrf_constant=rrf_constant,
        exclude_item_ids=cart_item_ids,
    )

    if len(recommendations) < k:
        fallback = recommend_with_popularity(
            popular_items=popular_items,
            cart_item_ids=cart_item_ids,
            k=max(k * 3, 50),
        )

        for item_id in fallback:
            if item_id in recommendations:
                continue

            recommendations.append(item_id)

            if len(recommendations) == k:
                break

    return recommendations


def evaluate_recommender(
    recommend_function: Callable[[list[int], int], list[int]],
    sampled_sessions: list[list[int]],
    item_catalog_ids: np.ndarray,
    popular_items: list[int],
    k: int,
) -> dict[str, float | int]:
    if not sampled_sessions:
        return {
            f"hit@{k}": 0.0,
            f"ndcg@{k}": 0.0,
            f"coverage@{k}": 0.0,
            "popularity_bias_top_1pct": 0.0,
            "n_sessions": 0,
        }

    top_one_percent_count = max(1, int(np.ceil(len(popular_items) * 0.01))) if popular_items else 0
    top_one_percent_items = set(popular_items[:top_one_percent_count])

    hit_values: list[float] = []
    ndcg_values: list[float] = []
    recommended_items_union: set[int] = set()
    total_recommendations = 0
    top_popular_recommendations = 0

    for session_item_ids in sampled_sessions:
        cart_item_ids = session_item_ids[:-1]
        target_item_id = int(session_item_ids[-1])

        recommended_item_ids = recommend_function(cart_item_ids, k)
        recommended_item_ids = list(dict.fromkeys(recommended_item_ids))[:k]

        recommended_items_union.update(recommended_item_ids)
        total_recommendations += len(recommended_item_ids)
        top_popular_recommendations += sum(
            1 for item_id in recommended_item_ids if item_id in top_one_percent_items
        )

        if target_item_id in recommended_item_ids:
            target_rank = recommended_item_ids.index(target_item_id) + 1
            hit_values.append(1.0)
            ndcg_values.append(1.0 / float(np.log2(target_rank + 1)))
        else:
            hit_values.append(0.0)
            ndcg_values.append(0.0)

    coverage = len(recommended_items_union) / len(item_catalog_ids) if len(item_catalog_ids) > 0 else 0.0
    popularity_bias = (
        top_popular_recommendations / total_recommendations
        if total_recommendations > 0
        else 0.0
    )

    return {
        f"hit@{k}": float(np.mean(hit_values)),
        f"ndcg@{k}": float(np.mean(ndcg_values)),
        f"coverage@{k}": float(coverage),
        "popularity_bias_top_1pct": float(popularity_bias),
        "n_sessions": len(sampled_sessions),
    }