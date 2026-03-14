from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np


class CartRecommender:
    def __init__(
        self,
        item_factors: np.ndarray,
        item_ids: np.ndarray,
        popular_items: list[int],
    ):
        self.item_factors = item_factors.astype(np.float32)
        self.item_ids = item_ids
        self.popular_items = popular_items
        self._item_id_to_idx: dict[int, int] = {iid: i for i, iid in enumerate(item_ids)}

        norms = np.linalg.norm(self.item_factors, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        self._normed_factors = self.item_factors / norms

    @classmethod
    def load(cls, artifact_path: Path) -> "CartRecommender":
        artifact = joblib.load(artifact_path)
        return cls(
            item_factors=artifact["item_factors"],
            item_ids=artifact["item_ids"],
            popular_items=artifact["popular_items"],
        )

    def recommend(self, cart_item_ids: list[int], n: int, exclude_item_ids: set[int] | None = None) -> list[int]:
        exclude = set(cart_item_ids)
        if exclude_item_ids:
            exclude.update(exclude_item_ids)

        known_idxs = [self._item_id_to_idx[iid] for iid in cart_item_ids if iid in self._item_id_to_idx]

        if not known_idxs:
            return self._popular_fallback(exclude, n)

        cart_vector = self._normed_factors[known_idxs].mean(axis=0)
        query_norm = np.linalg.norm(cart_vector)
        if query_norm < 1e-9:
            return self._popular_fallback(exclude, n)

        query = cart_vector / query_norm
        scores = self._normed_factors @ query

        exclude_idxs = {self._item_id_to_idx[iid] for iid in exclude if iid in self._item_id_to_idx}
        scores[list(exclude_idxs)] = -np.inf

        top_idxs = np.argpartition(scores, -n)[-n:] if len(scores) >= n else np.arange(len(scores))
        top_idxs = top_idxs[np.argsort(scores[top_idxs])[::-1]]

        recommendations = [int(self.item_ids[i]) for i in top_idxs if scores[i] > -np.inf]
        return recommendations[:n]

    def _popular_fallback(self, exclude: set[int], n: int) -> list[int]:
        return [iid for iid in self.popular_items if iid not in exclude][:n]
