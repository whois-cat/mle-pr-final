import pytest

from cart_driven_recsys import api


def test_health_returns_ok(client, monkeypatch, artifact):
    monkeypatch.setattr(api, "get_artifact", lambda: artifact)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metadata_returns_expected_payload(client, monkeypatch, artifact):
    monkeypatch.setattr(api, "get_artifact", lambda: artifact)

    response = client.get("/metadata")
    assert response.status_code == 200

    payload = response.json()

    expected = {
        "model_type": "hybrid_als_covisit",
        "n_items": 4,
        "n_users": 2,
        "cutoff_date": "2026-03-14",
        "als_factors": 128,
        "covisit_top_neighbors": 50,
        "als_weight": 0.7,
        "covisit_weight": 0.3,
        "rrf_constant": 60,
    }

    for key, value in expected.items():
        assert payload[key] == value, f"Mismatch in {key}"

    assert isinstance(payload["artifact_path"], str)
    assert payload["artifact_path"]


def test_metrics_endpoint_returns_prometheus_format(client, monkeypatch, artifact):
    monkeypatch.setattr(api, "get_artifact", lambda: artifact)

    # Warm-up call to generate some metrics
    client.get("/health")

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers.get("content-type", "")
    assert "http_requests_total" in response.text


def test_recommend_cart_returns_top_k_items(client, monkeypatch, artifact):
    monkeypatch.setattr(api, "get_artifact", lambda: artifact)
    monkeypatch.setattr(api, "recommend_with_hybrid", lambda **_: [4, 3, 2])

    response = client.post(
        "/recommendations/cart",
        json={"item_ids": [1], "k": 3},
    )

    assert response.status_code == 200
    assert response.json() == {
        "item_ids": [4, 3, 2],
        "unknown_item_ids": [],
    }


def test_recommend_cart_correctly_reports_unknown_items(client, monkeypatch, artifact):
    monkeypatch.setattr(api, "get_artifact", lambda: artifact)
    monkeypatch.setattr(api, "recommend_with_hybrid", lambda **_: [4, 3])

    response = client.post(
        "/recommendations/cart",
        json={"item_ids": [1, 999, 1, 888], "k": 2},
    )

    assert response.status_code == 200
    assert response.json() == {
        "item_ids": [4, 3],
        "unknown_item_ids": [999, 888],
    }


def test_recommend_cart_removes_duplicates_before_passing_to_model(client, monkeypatch, artifact):
    captured = {}

    def fake_recommend(**kwargs):
        captured.update(kwargs)
        return [4, 3]

    monkeypatch.setattr(api, "get_artifact", lambda: artifact)
    monkeypatch.setattr(api, "recommend_with_hybrid", fake_recommend)

    client.post(
        "/recommendations/cart",
        json={"item_ids": [1, 2, 1, 2, 3], "k": 2},
    )

    assert captured["cart_item_ids"] == [1, 2, 3]


@pytest.mark.parametrize("invalid_k", [0, -5, 101, 1000])
def test_recommend_cart_rejects_invalid_k_values(client, monkeypatch, artifact, invalid_k):
    monkeypatch.setattr(api, "get_artifact", lambda: artifact)

    response = client.post(
        "/recommendations/cart",
        json={"item_ids": [1, 2], "k": invalid_k},
    )

    assert response.status_code == 422


def test_recommend_cart_returns_500_when_artifact_cannot_be_loaded(client, monkeypatch):
    def broken_artifact():
        raise api.HTTPException(status_code=500, detail="failed to load model artifact")

    monkeypatch.setattr(api, "get_artifact", broken_artifact)

    response = client.post(
        "/recommendations/cart",
        json={"item_ids": [1], "k": 2},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "failed to load model artifact"}


def test_recommend_cart_returns_500_on_recommendation_failure(client, monkeypatch, artifact):
    monkeypatch.setattr(api, "get_artifact", lambda: artifact)

    def broken_recommend(**_):
        raise RuntimeError("boom")

    monkeypatch.setattr(api, "recommend_with_hybrid", broken_recommend)

    response = client.post(
        "/recommendations/cart",
        json={"item_ids": [1], "k": 2},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "failed to generate recommendations"}