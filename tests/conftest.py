from __future__ import annotations

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient


class FakeRecommender:
    def recommend(
        self,
        user_id: int | None = None,
        cart_item_ids: list[int] | None = None,
        k: int = 10,
        exclude_item_ids: list[int] | None = None,
    ) -> list[int]:
        return [101, 102, 103][:k]


@pytest.fixture
def client():
    with patch("cart_driven_recsys.api.load_artifact", return_value={"fake": "artifact"}), patch(
        "cart_driven_recsys.api.CartRecommender",
        return_value=FakeRecommender(),
    ):
        from cart_driven_recsys.api import app

        with TestClient(app) as test_client:
            yield test_client