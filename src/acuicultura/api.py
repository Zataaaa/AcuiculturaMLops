"""Servicio de inferencia (FastAPI).

Uso:  uvicorn acuicultura.api:app --host 0.0.0.0 --port 8000
"""
import json
import os
import time
from datetime import datetime

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .features import DO_THRESHOLD, EARLY_MARGIN, HORIZON, make_features

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
METRICS_PATH = os.getenv("METRICS_PATH", "artifacts/metrics.json")
PRED_LOG = os.getenv("PRED_LOG", "artifacts/predictions.jsonl")
MIN_READINGS = 7  # 6 para ventanas móviles + 1 para la diferencia

app = FastAPI(title="Riesgo de baja de oxígeno - Acuicultura", version=__version__)
_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


class Reading(BaseModel):
    timestamp: datetime
    temperature_c: float = Field(ge=10, le=40)
    ph: float = Field(ge=4, le=11)
    dissolved_oxygen_mgl: float = Field(ge=0, le=20)


class PredictRequest(BaseModel):
    readings: list[Reading] = Field(min_length=MIN_READINGS, description="Lecturas horarias recientes")


class PredictResponse(BaseModel):
    risk_score: float
    alert: bool
    horizon_h: int
    do_threshold_mgl: float
    model_version: str
    status: str  # 'modelo' o 'oxigeno_ya_bajo' (ya está por debajo del umbral: alerta directa)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    try:
        with open(METRICS_PATH, encoding="utf-8") as f:
            m = json.load(f)
    except FileNotFoundError:
        raise HTTPException(503, "metrics.json no disponible")
    return {"best_model": m["best_model"], "do_threshold_mgl": m["do_threshold_mgl"],
            "horizon_h": m["horizon_h"], "test": m["results"][m["best_model"]]["test"],
            "api_version": __version__}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        b = get_bundle()
    except FileNotFoundError:
        raise HTTPException(503, "Modelo no disponible")
    df = pd.DataFrame([r.model_dump() for r in req.readings]).sort_values("timestamp").reset_index(drop=True)
    x = make_features(df).iloc[[-1]]
    if x.isna().any(axis=1).iloc[0]:
        raise HTTPException(422, "No se pudieron calcular las features (¿lecturas no consecutivas?)")
    if x["do"].iloc[0] < DO_THRESHOLD + EARLY_MARGIN:  # ya está bajo: no hace falta modelo
        score, status, alert = 1.0, "oxigeno_ya_bajo", True
    else:
        score = float(b["model"].predict_proba(x)[:, 1][0])
        status, alert = "modelo", score >= b["threshold"]
    out = PredictResponse(risk_score=score, alert=alert, horizon_h=HORIZON,
                          do_threshold_mgl=DO_THRESHOLD, model_version=__version__, status=status)
    try:  # log de predicciones: base para monitoreo de drift y desempeño
        os.makedirs(os.path.dirname(PRED_LOG) or ".", exist_ok=True)
        with open(PRED_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), **x.iloc[0].to_dict(), **out.model_dump()}) + "\n")
    except OSError:
        pass
    return out
