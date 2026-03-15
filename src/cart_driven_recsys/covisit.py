from __future__ import annotations

from collections import defaultdict
import pandas as pd
from cart_driven_recsys.recommenders import build_sessions_from_events


def build_covisit_index(
    train_events: pd.DataFrame,
    gap_hours: int,
    top_neighbors_per_item: int = 50,
) -> dict[int, list[tuple[int, float]]]:
    sessions = build_sessions_from_events(train_events, gap_hours=gap_hours)
    pair_scores: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))

    for session_item_ids in sessions:
        unique_session_item_ids = list(dict.fromkeys(int(item_id) for item_id in session_item_ids))

        if len(unique_session_item_ids) < 2:
            continue

        for left_position in range(len(unique_session_item_ids) - 1):
            left_item_id = unique_session_item_ids[left_position]

            for right_position in range(left_position + 1, len(unique_session_item_ids)):
                right_item_id = unique_session_item_ids[right_position]
                score_increment = 1.0 / (right_position - left_position)

                pair_scores[left_item_id][right_item_id] += score_increment
                pair_scores[right_item_id][left_item_id] += score_increment

    covisit_index: dict[int, list[tuple[int, float]]] = {}

    for source_item_id, neighbor_scores in pair_scores.items():
        covisit_index[source_item_id] = sorted(
            neighbor_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[:top_neighbors_per_item]

    return covisit_index