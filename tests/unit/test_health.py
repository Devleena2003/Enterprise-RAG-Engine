from fastapi.testclient import TestClient

from rag_engine.main import create_app


def test_health_live() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.get("/health/live")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "X-Request-ID" in response.headers


def test_health_ready_returns_503_when_db_down(monkeypatch) -> None:
    # Patch where health.py uses the function
    monkeypatch.setattr(
        "rag_engine.api.health.check_db_readiness",
        lambda: (False, "database_unreachable"),
    )
    app = create_app()
    client = TestClient(app)

    response = client.get("/health/ready", headers={"X-Request-ID": "req_test_123"})

    assert response.status_code == 503
    assert response.headers["X-Request-ID"] == "req_test_123"
    assert response.json()["status"] == "degraded"
    assert response.json()["checks"]["database"] == "database_unreachable"


def test_health_ready_returns_200_when_db_up(monkeypatch) -> None:
    monkeypatch.setattr(
        "rag_engine.api.health.check_db_readiness",
        lambda: (True, "ok"),
    )
    app = create_app()
    client = TestClient(app)

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"