import pytest

from acuicultura.data import validate


def test_valid_data_passes(df):
    out = validate(df)
    assert len(out) == len(df)


def test_missing_column_fails(df):
    with pytest.raises(ValueError, match="Faltan columnas"):
        validate(df.drop(columns=["ph"]))


def test_nulls_fail(df):
    df.loc[3, "temperature_c"] = None
    with pytest.raises(ValueError, match="nulos"):
        validate(df)


def test_out_of_range_fails(df):
    df.loc[5, "ph"] = 14.0
    with pytest.raises(ValueError, match="fuera de rango"):
        validate(df)


def test_duplicate_timestamps_fail(df):
    df.loc[1, "timestamp"] = df.loc[0, "timestamp"]
    with pytest.raises(ValueError, match="duplicados"):
        validate(df)


def test_sorted_output(df):
    out = validate(df.sample(frac=1, random_state=1))
    assert out["timestamp"].is_monotonic_increasing
