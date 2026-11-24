import json

import numpy as np

from acuicultura.monitoring import drift_report, psi
from acuicultura.train import pick_threshold, run, temporal_split


def test_temporal_split_keeps_order_and_sizes():
    X, y = np.arange(100), np.arange(100)
    (a, _), (b, _), (c, _) = temporal_split(X, y)
    assert len(a) == 70 and len(b) == 15 and len(c) == 15
    assert a.max() < b.min() and b.max() < c.min()  # sin fuga temporal


def test_pick_threshold_meets_recall():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.3, 0.6, 0.2, 0.1, 0.05])
    t = pick_threshold(y, p, target_recall=0.75)
    assert ((p >= t) & (y == 1)).sum() / 4 >= 0.75


def test_psi_same_distribution_is_small_and_shift_is_large():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.1
    assert psi(a, rng.normal(2, 1, 5000)) > 0.2


def test_drift_report_flags_shift():
    import pandas as pd
    rng = np.random.default_rng(2)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 2000)})
    new = pd.DataFrame({"x": rng.normal(3, 1, 2000)})
    assert drift_report(ref, new, ["x"])["drift"] is True


def test_training_pipeline_end_to_end(df, tmp_path):
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    meta = run(str(csv), str(tmp_path / "out"))
    assert (tmp_path / "out" / "model.joblib").exists()
    saved = json.loads((tmp_path / "out" / "metrics.json").read_text(encoding="utf-8"))
    assert saved["best_model"] in {"regresion_logistica", "gradient_boosting"}
    assert set(meta["results"]) == {"regla_oxigeno_actual", "regresion_logistica", "gradient_boosting"}
