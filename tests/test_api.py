import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

SAMPLE = json.loads((Path(__file__).resolve().parents[1] / "sample_input.json").read_text())


@pytest.fixture()
def client(model_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    from src.api import app
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_predict_valid_payload(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert 0.0 <= body["probability_disease"] <= 1.0
    assert body["risk_level"] in {"low", "moderate", "high"}
    assert "X-Request-ID" in r.headers


def test_optional_fields_can_be_omitted(client):
    payload = {k: v for k, v in SAMPLE.items() if k not in ("ca", "thal")}
    assert client.post("/predict", json=payload).status_code == 200


def test_out_of_range_value_rejected(client):
    assert client.post("/predict", json={**SAMPLE, "age": 500}).status_code == 422


def test_missing_required_field_rejected(client):
    payload = {k: v for k, v in SAMPLE.items() if k != "chol"}
    assert client.post("/predict", json=payload).status_code == 422


def test_metrics_endpoint_exposes_counters(client):
    client.post("/predict", json=SAMPLE)
    text = client.get("/metrics").text
    assert "http_requests_total" in text and "predictions_total" in text


def test_model_info(client):
    assert client.get("/model-info").status_code == 200
