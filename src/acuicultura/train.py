"""Entrenamiento y comparación de modelos con split temporal (70/15/15)."""
import argparse
import json
import os

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_csv
from .features import DO_THRESHOLD, FEATURE_COLS, HORIZON, build_dataset

TARGET_RECALL = 0.70  # el costo de no avisar es mayor que el de una falsa alarma


class OxygenRule:
    """Baseline sin ML: mientras más bajo el oxígeno actual, más riesgo. Score = -oxígeno."""

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        s = 1.0 / (1.0 + X["do"].to_numpy(dtype=float))  # monótono y positivo
        return np.column_stack([1 - s, s])


def temporal_split(X, y):
    n = len(X)
    a, b = int(n * 0.70), int(n * 0.85)
    return (X[:a], y[:a]), (X[a:b], y[a:b]), (X[b:], y[b:])


def pick_threshold(y_val, p_val, target_recall=TARGET_RECALL):
    """Umbral más alto que todavía cumple el recall objetivo en validación."""
    best = None
    for t in np.unique(p_val):
        if recall_score(y_val, p_val >= t) >= target_recall:
            best = float(t) if best is None else max(best, float(t))
    return float(np.min(p_val)) if best is None else best


def evaluate(y, p, thr):
    pred = (p >= thr).astype(int)
    return {
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "alert_rate": float(pred.mean()),
        "prevalence": float(np.mean(y)),
        "threshold": float(thr),
    }


def candidates():
    return {
        "regla_oxigeno_actual": OxygenRule(),
        "regresion_logistica": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_depth=3, class_weight="balanced", random_state=42),
    }


def run(data_path: str, out_dir: str):
    df = load_csv(data_path)
    X, y = build_dataset(df)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = temporal_split(X, y)
    results, models = {}, {}
    for name, model in candidates().items():
        model.fit(Xtr, ytr)
        thr = pick_threshold(yva, model.predict_proba(Xva)[:, 1])
        results[name] = {
            "val": evaluate(yva, model.predict_proba(Xva)[:, 1], thr),
            "test": evaluate(yte, model.predict_proba(Xte)[:, 1], thr),
        }
        models[name] = (model, thr)
    ml = {k: v for k, v in results.items() if k != "regla_oxigeno_actual"}
    best = max(ml, key=lambda k: results[k]["val"]["pr_auc"])
    meta = {
        "best_model": best, "do_threshold_mgl": DO_THRESHOLD, "horizon_h": HORIZON,
        "features": FEATURE_COLS, "n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte),
        "alert_threshold": models[best][1], "results": results,
        # Perfil de referencia para detectar drift (media/desv. de entrenamiento)
        "reference": {c: {"mean": float(Xtr[c].mean()), "std": float(Xtr[c].std())} for c in FEATURE_COLS},
    }
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump({"model": models[best][0], "threshold": models[best][1], "features": FEATURE_COLS},
                os.path.join(out_dir, "model.joblib"))
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    for name, r in results.items():  # formato que CloudWatch/SageMaker puede capturar con regex
        t = r["test"]
        print(f"[{name}] test_recall={t['recall']:.4f} test_precision={t['precision']:.4f} "
              f"test_pr_auc={t['pr_auc']:.4f} test_roc_auc={t['roc_auc']:.4f}")
    print(f"best_model={best}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/hourly_ponds.csv")
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args()
    run(a.data, a.out)
