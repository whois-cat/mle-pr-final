def test_health_returns_ok(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_returns_ok_when_model_is_loaded(client) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_recommendations_returns_items(client) -> None:
    response = client.get("/recommendations", params={"user_id": 123, "k": 2})
    assert response.status_code == 200
    assert response.json()["items"] == [101, 102]


def test_recommendations_validates_k(client) -> None:
    response = client.get("/recommendations", params={"user_id": 123, "k": 0})
    assert response.status_code in {400, 422}


def test_metrics_endpoint_returns_prometheus_text(client) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]