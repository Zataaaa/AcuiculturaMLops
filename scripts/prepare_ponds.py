"""Limpia los CSV crudos del dataset de estanques (Udanor et al., Nsukka, Nigeria) y genera data/hourly_ponds.csv.

Uso:  python scripts/prepare_ponds.py <carpeta_con_IoTPond*.csv> [salida.csv]

Pasos: unifica nombres de columnas, descarta lecturas imposibles (sensor desconectado = -127 C, etc.),
agrupa por hora con la mediana y conserva solo las horas con al menos 20 lecturas validas.
Los estanques 5, 8, 10, 11 y 12 se excluyen (faltan, no se pudieron leer o son muy cortos).
"""
import glob
import os
import re
import sys

import pandas as pd

PONDS = [1, 2, 3, 4, 6, 7, 9]
MIN_READINGS_PER_HOUR = 20
LIMITS = {"temperature_c": (15, 40), "ph": (4, 11), "dissolved_oxygen_mgl": (0, 20)}


def _name(col: str):
    c = col.lower()
    if c.startswith("created_at"):
        return "timestamp"
    if "temp" in c:
        return "temperature_c"
    if "oxygen" in c:
        return "dissolved_oxygen_mgl"
    if c == "ph":
        return "ph"
    return None


def load_raw(path: str) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    keep = {}
    for c in d.columns:
        n = _name(c)
        if n and n not in keep.values():
            keep[c] = n
    d = d[list(keep)].rename(columns=keep)
    d["timestamp"] = pd.to_datetime(d["timestamp"].astype(str).str.replace(r"\s*(CET|UTC|CEST)", "", regex=True),
                                    errors="coerce")
    for c in d.columns:
        if c != "timestamp":
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["timestamp"]).sort_values("timestamp")


def to_hourly(d: pd.DataFrame) -> pd.DataFrame:
    for c, (lo, hi) in LIMITS.items():
        d = d[d[c].between(lo, hi)]
    d = d.set_index("timestamp")
    hourly = d.resample("h").median()
    n = d["dissolved_oxygen_mgl"].resample("h").count()
    return hourly[n >= MIN_READINGS_PER_HOUR].dropna().reset_index()


def main(folder: str, out: str):
    frames = []
    for path in glob.glob(os.path.join(folder, "*.csv")):
        m = re.findall(r"ond(\d+)", os.path.basename(path), re.IGNORECASE)
        if m and int(m[-1]) in PONDS:
            h = to_hourly(load_raw(path))
            h.insert(1, "pond", int(m[-1]))
            frames.append(h)
    df = pd.concat(frames).sort_values(["pond", "timestamp"]).reset_index(drop=True)
    df.round({"temperature_c": 3, "ph": 3, "dissolved_oxygen_mgl": 3}).to_csv(out, index=False)
    print(df.groupby("pond").size().to_string(), f"\ntotal {len(df)} filas -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "data/hourly_ponds.csv")
