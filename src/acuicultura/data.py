"""Carga y validación de datos."""
import pandas as pd

REQUIRED = ["timestamp", "temperature_c", "dissolved_oxygen_mgl", "ph"]
# Rangos físicamente plausibles para un estanque (los sensores a veces reportan -127 °C al desconectarse)
RANGES = {
    "temperature_c": (10.0, 40.0),
    "dissolved_oxygen_mgl": (0.0, 20.0),
    "ph": (4.0, 11.0),
}


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Valida el dataframe y lo regresa ordenado (por estanque y tiempo). Lanza ValueError si algo está mal.

    La columna 'pond' es opcional; si no existe se asume un solo estanque.
    """
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")
    if df[REQUIRED].isna().any().any():
        raise ValueError("Hay valores nulos en las columnas requeridas")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "pond" not in df.columns:
        df["pond"] = 0
    if df.duplicated(["pond", "timestamp"]).any():
        raise ValueError("Hay timestamps duplicados")
    for col, (lo, hi) in RANGES.items():
        bad = ((df[col] < lo) | (df[col] > hi)).sum()
        if bad:
            raise ValueError(f"{col}: {bad} valores fuera de rango [{lo}, {hi}]")
    return df.sort_values(["pond", "timestamp"]).reset_index(drop=True)


def load_csv(path: str) -> pd.DataFrame:
    return validate(pd.read_csv(path))
