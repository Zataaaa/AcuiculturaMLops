"""Construcción de features y etiqueta de riesgo.

Etiqueta: 1 si el oxígeno disuelto mínimo de las próximas HORIZON horas cae por debajo de DO_THRESHOLD.
Alerta temprana: el modelo solo se entrena y evalúa en horas en que el oxígeno todavía está normal
(>= DO_THRESHOLD + EARLY_MARGIN). Si ya está bajo no hay nada que predecir: la regla simple basta.
Features: temperatura, pH y oxígeno actuales, sus estadísticas de las últimas 6 horas y la hora del día.
Solo usa el pasado. Con huecos en los datos, las ventanas se calculan sobre una malla horaria completa
y se descartan las filas que quedan con valores faltantes.
"""
import numpy as np
import pandas as pd

DO_THRESHOLD = 4.0  # mg/L, umbral de estrés para peces de estanque (ver README: pendiente de respaldar con fuente)
EARLY_MARGIN = 0.5  # solo se predice cuando el oxígeno actual >= DO_THRESHOLD + EARLY_MARGIN
HORIZON = 3  # horas hacia adelante
BASE = {"temperature_c": "temp", "ph": "ph", "dissolved_oxygen_mgl": "do"}

FEATURE_COLS = (
    ["temp", "ph", "do", "hour_sin", "hour_cos"]
    + [f"{n}_{s}" for n in BASE.values() for s in ("mean6", "std6", "diff1")]
)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features a partir de lecturas horarias consecutivas y ordenadas de un estanque (solo usa el pasado)."""
    X = pd.DataFrame(index=df.index)
    hour = df["timestamp"].dt.hour
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    for col, n in BASE.items():
        X[n] = df[col]
        X[f"{n}_mean6"] = df[col].rolling(6).mean()
        X[f"{n}_std6"] = df[col].rolling(6).std()
        X[f"{n}_diff1"] = df[col].diff()
    return X[FEATURE_COLS]


def make_labels(df: pd.DataFrame, thr: float = DO_THRESHOLD, horizon: int = HORIZON) -> pd.Series:
    fut = pd.concat([df["dissolved_oxygen_mgl"].shift(-k) for k in range(1, horizon + 1)], axis=1)
    y = (fut.min(axis=1) < thr).astype(float)
    y[fut.isna().any(axis=1)] = np.nan
    return y


def _hourly_grid(g: pd.DataFrame) -> pd.DataFrame:
    g = g.set_index("timestamp")
    g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="h"))
    g.index.name = "timestamp"
    return g.reset_index()


def build_dataset(df: pd.DataFrame, early_only: bool = True):
    """(X, y) de todos los estanques, ordenados por tiempo (el split temporal es global, sin mezclar futuro)."""
    if "pond" not in df.columns:
        df = df.assign(pond=0)
    Xs, ys, ts = [], [], []
    for _, g in df.groupby("pond"):
        g = _hourly_grid(g)
        X, y = make_features(g), make_labels(g)
        mask = X.notna().all(axis=1) & y.notna()
        if early_only:
            mask &= X["do"] >= DO_THRESHOLD + EARLY_MARGIN
        Xs.append(X[mask]), ys.append(y[mask]), ts.append(g.loc[mask, "timestamp"])
    X, y, t = pd.concat(Xs), pd.concat(ys), pd.concat(ts)
    order = np.argsort(t.to_numpy(), kind="stable")
    return X.iloc[order].reset_index(drop=True), y.iloc[order].astype(int).reset_index(drop=True)
