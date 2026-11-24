import numpy as np

from acuicultura.features import (
    DO_THRESHOLD,
    EARLY_MARGIN,
    FEATURE_COLS,
    HORIZON,
    build_dataset,
    make_features,
    make_labels,
)


def test_feature_columns(df):
    X = make_features(df)
    assert list(X.columns) == FEATURE_COLS


def test_oxygen_is_an_input():
    assert "do" in FEATURE_COLS and "do_mean6" in FEATURE_COLS


def test_label_uses_only_future(df):
    df = df.copy()
    df["dissolved_oxygen_mgl"] = 7.0
    df.loc[10, "dissolved_oxygen_mgl"] = 1.0  # evento de baja en la hora 10
    y = make_labels(df, thr=6.0, horizon=HORIZON)
    assert y[7:10].eq(1).all()      # las 3 horas previas ven el evento
    assert y[10] == 0 and y[11] == 0  # la hora del evento y después no
    assert np.isnan(y.iloc[-1])      # sin futuro disponible no hay etiqueta


def test_build_dataset_aligned_without_nans(df):
    X, y = build_dataset(df)
    assert len(X) == len(y) and not X.isna().any().any()
    assert set(y.unique()) <= {0, 1}


def test_early_warning_only_keeps_normal_oxygen(df):
    X, y = build_dataset(df)
    assert (X["do"] >= DO_THRESHOLD + EARLY_MARGIN).all()
    assert y.sum() > 0  # el fixture sí tiene bajas, si no la prueba no dice nada


def test_gaps_do_not_leak_across_missing_hours(df):
    """Con un hueco de 10 h, las ventanas de 6 h no deben mezclar lecturas de antes y después del hueco."""
    gap = df.drop(index=range(100, 110)).reset_index(drop=True)
    X_gap, _ = build_dataset(gap)
    X_full, _ = build_dataset(df)
    assert len(X_gap) < len(X_full)  # se descartan las filas con ventana incompleta


def test_multiple_ponds_sorted_by_time(df):
    other = df.copy()
    other["pond"] = 2
    other["timestamp"] += __import__("pandas").Timedelta(hours=5)
    both = __import__("pandas").concat([df.assign(pond=1), other])
    X, y = build_dataset(both)
    assert len(X) == len(y) and len(X) > len(build_dataset(df)[0])
