import importlib

import pytest
from fastapi.testclient import TestClient

from acuicultura.train import run


@pytest.fixture
def client(df, tmp_path, monkeypatch):
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    run(str(csv), str(tmp_path))
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "model.joblib"))
    monkeypatch.setenv("METRICS_PATH", str(tmp_path / "metrics.json"))
    monkeypatch.setenv("PRED_LOG", str(tmp_path / "pred.jsonl"))
    from acuicultura import api
    importlib.reload(api)
    return TestClient(api.app)


def payload(df, n=8):
    rows = df.head(n)
    return {"readings": [
        {"timestamp": t.isoformat(), "temperature_c": a, "ph": b, "dissolved_oxygen_mgl": c}
        for t, a, b, c in zip(rows.timestamp, rows.temperature_c, rows.ph, rows.dissolved_oxygen_mgl)]}


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_ok(client, df):
    r = client.post("/predict", json=payload(df))
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["risk_score"] <= 1.0 and isinstance(body["alert"], bool)


def test_already_low_oxygen_alerts_without_model(client, df):
    p = payload(df)
    p["readings"][-1]["dissolved_oxygen_mgl"] = 2.0
    body = client.post("/predict", json=p).json()
    assert body["alert"] is True and body["status"] == "oxigeno_ya_bajo" and body["risk_score"] == 1.0


def test_predict_too_few_readings_422(client, df):
    assert client.post("/predict", json=payload(df, n=3)).status_code == 422


def test_predict_out_of_range_422(client, df):
    p = payload(df)
    p["readings"][0]["ph"] = 99
    assert client.post("/predict", json=p).status_code == 422


def test_model_info(client):
    assert client.get("/model-info").json()["horizon_h"] == 3
