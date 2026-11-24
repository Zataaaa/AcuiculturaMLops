import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def df():
    """Un estanque sintético de 600 h con ciclo diario de oxígeno (a veces baja de 4 mg/L)."""
    n = 600
    rng = np.random.default_rng(0)
    hour = np.arange(n) % 24
    do = 5.5 + 2.2 * np.sin(2 * np.pi * (hour - 8) / 24) + rng.normal(0, 0.5, n)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "temperature_c": rng.normal(27, 0.6, n),
        "dissolved_oxygen_mgl": np.clip(do, 0.3, 15),
        "ph": rng.normal(7.5, 0.2, n),
    })
