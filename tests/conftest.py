import numpy as np
import pytest
from fastapi.testclient import TestClient

from cart_driven_recsys import api


@pytest.fixture(autouse=True)
def clear_artifact_cache():
    api._load_artifact.cache_clear()
    yield
    api._load_artifact.cache_clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(api.app)


@pytest.fixture
def artifact() -> dict:
    return {
        "model_type": "hybrid_als_covisit",
        "als_model": object(),
        "item_ids": np.array([1, 2, 3, 4], dtype=np.int64),
        "item_id_to_index": {1: 0, 2: 1, 3: 2, 4: 3},
        "item_id_set": {1, 2, 3, 4},
        "popular_items": [4, 3, 2, 1],
        "covisit_index": {},
        "hybrid_params": {
            "als_weight": 0.7,
            "covisit_weight": 0.3,
            "rrf_constant": 60,
        },
        "train_meta": {
            "n_items": 4,
            "n_users": 2,
            "cutoff_date": "2026-03-14",
            "als_factors": 128,
            "covisit_top_neighbors": 50,
        },
    }