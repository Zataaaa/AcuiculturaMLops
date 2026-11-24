import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lambdas"))
pytest.importorskip("boto3")
if not (Path(__file__).resolve().parents[1] / "lambdas" / "model.json").exists():
    pytest.skip("falta lambdas/model.json (ver scripts/export_lambda_model.py)", allow_module_level=True)
import handler

from acuicultura.features import FEATURE_COLS, make_features


class FakeClient:
    def __getattr__(self, name):
        return lambda *a, **k: None


@pytest.fixture(autouse=True)
def no_aws(monkeypatch):
    monkeypatch.setattr(handler.boto3, "client", lambda *a, **k: FakeClient())


def readings(df, n=8):
    rows = df.head(n)
    return [{"timestamp": t.isoformat(), "temperature_c": a, "ph": b, "dissolved_oxygen_mgl": c}
            for t, a, b, c in zip(rows.timestamp, rows.temperature_c, rows.ph, rows.dissolved_oxygen_mgl)]


def test_features_match_training_pipeline(df):
    f = handler.features(readings(df))
    ref = make_features(df.head(8)).iloc[-1]
    assert max(abs(f[c] - ref[c]) for c in FEATURE_COLS) < 1e-9


def test_predict_ok_and_422(df):
    ev = {"requestContext": {"http": {"method": "POST"}}, "rawPath": "/predict"}
    ok = handler.api({**ev, "body": json.dumps({"readings": readings(df)})})
    assert ok["statusCode"] == 200 and 0 <= json.loads(ok["body"])["risk_score"] <= 1
    bad = handler.api({**ev, "body": json.dumps({"readings": readings(df, 3)})})
    assert bad["statusCode"] == 422


def test_exported_trees_match_scikit_learn(df, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from export_lambda_model import export_trees

    from acuicultura.features import build_dataset
    from acuicultura.train import candidates
    X, y = build_dataset(df)
    model = candidates()["gradient_boosting"].fit(X, y)
    monkeypatch.setattr(handler, "M", {**handler.M, **export_trees(model)})
    for _, row in X.head(20).iterrows():
        assert abs(handler.score(row.to_dict()) - model.predict_proba(row.to_frame().T)[0, 1]) < 1e-6


def test_already_low_oxygen_alerts(df):
    ev = {"requestContext": {"http": {"method": "POST"}}, "rawPath": "/predict"}
    rs = readings(df)
    rs[-1]["dissolved_oxygen_mgl"] = 1.5
    out = json.loads(handler.api({**ev, "body": json.dumps({"readings": rs})})["body"])
    assert out["alert"] is True and out["status"] == "oxigeno_ya_bajo"


def test_psi_detects_shift():
    dec = handler.M["ref_deciles"]["temp"]
    normal = [dec[i] + 0.001 for i in range(10)] * 10
    shifted = [dec[-1]] * 100
    assert handler._psi(normal, dec) < 0.2 < handler._psi(shifted, dec)
