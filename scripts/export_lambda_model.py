"""Exporta el gradient boosting entrenado a JSON (arboles + perfil de referencia) para la Lambda sin scikit-learn.

Uso:  PYTHONPATH=src python scripts/export_lambda_model.py artifacts lambdas/model.json data/hourly_ponds.csv
"""
import json
import sys

import joblib
import numpy as np

from acuicultura.data import load_csv
from acuicultura.features import DO_THRESHOLD, EARLY_MARGIN, FEATURE_COLS, HORIZON, build_dataset
from acuicultura.train import temporal_split


def export_trees(model):
    trees = []
    for stage in model._predictors:
        nodes = stage[0].nodes
        trees.append([[-1 if n["is_leaf"] else int(n["feature_idx"]), float(n["num_threshold"]),
                       int(n["left"]), int(n["right"]), float(n["value"])] for n in nodes])
    return {"baseline": float(np.ravel(model._baseline_prediction)[0]), "trees": trees}


def main(art, out, data):
    b = joblib.load(f"{art}/model.joblib")
    with open(f"{art}/metrics.json", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["best_model"] == "gradient_boosting", "el exportador de la Lambda solo soporta gradient boosting"
    X, y = build_dataset(load_csv(data))
    (Xtr, _), _, _ = temporal_split(X, y)
    deciles = {c: [round(float(v), 6) for v in np.quantile(Xtr[c], np.linspace(0, 1, 11))] for c in FEATURE_COLS}
    props = {}
    for c in ("hour_sin", "hour_cos"):  # variables discretas: proporciones reales por intervalo (evita falsos positivos)
        edges = deciles[c][1:-1]
        idx = np.searchsorted(edges, Xtr[c].to_numpy(), side="left")
        props[c] = [round(float(v), 6) for v in np.bincount(idx, minlength=len(edges) + 1) / len(Xtr)]
    m = {"model": "gradient_boosting", "features": FEATURE_COLS, "threshold": b["threshold"],
         "do_threshold_mgl": DO_THRESHOLD, "early_margin": EARLY_MARGIN, "horizon_h": HORIZON,
         "test": meta["results"]["gradient_boosting"]["test"], "ref_deciles": deciles, "ref_props": props,
         **export_trees(b["model"])}
    # verificacion contra scikit-learn
    sample = Xtr.iloc[:300]
    ref = b["model"].predict_proba(sample)[:, 1]
    got = []
    for _, r in sample.iterrows():
        z = m["baseline"]
        for tr in m["trees"]:
            i = 0
            while tr[i][0] >= 0:
                i = tr[i][2] if r[FEATURE_COLS[tr[i][0]]] <= tr[i][1] else tr[i][3]
            z += tr[i][4]
        got.append(1 / (1 + np.exp(-z)))
    err = float(np.max(np.abs(np.array(got) - ref)))
    print("max |diff| vs scikit-learn:", err)
    assert err < 1e-6
    with open(out, "w", encoding="utf-8") as f:
        json.dump(m, f, separators=(",", ":"))
    print("escrito", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
