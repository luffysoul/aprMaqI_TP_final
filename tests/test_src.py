"""Tests del código compartido en src/.

Cada test cubre un invariante crítico del TP. Todos fueron validados con la
regla de sabotaje: se inyectó temporalmente el bug que dicen cubrir y se
verificó que el test falla (evidencia en el historial de la Fase 1).
"""

import numpy as np
import pandas as pd
import pytest

from src.config import COLS_FILTRO_IQR, FEATURES_FINALES
from src.evaluacion import registrar
from src.preprocesamiento import filtrar_outliers_train, split_estratificado


def _train_sintetico(n: int = 101) -> tuple[pd.DataFrame, pd.Series]:
    """Train sintético: columnas de filtro constantes (IQR=0) salvo una,
    de modo que los límites IQR quedan totalmente determinados."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({col: np.full(n, 5.0) for col in COLS_FILTRO_IQR})
    # num_medications: uniforme 10..20 -> IQR acotado, límites conocidos
    X["num_medications"] = rng.integers(10, 21, size=n).astype(float)
    X["age"] = pd.Categorical(["[50-60)"] * n)
    for c in ["diag_1", "diag_2", "diag_3"]:
        X[c] = np.int16(7)
    y = pd.Series(np.full(n, 4.0), name="time_in_hospital")
    return X, y


def test_filtro_iqr_elimina_outliers_con_umbral_1_5():
    """El filtro debe usar exactamente [Q1-1.5*IQR, Q3+1.5*IQR] del train."""
    X, y = _train_sintetico()
    q1, q3 = X["num_medications"].quantile(0.25), X["num_medications"].quantile(0.75)
    iqr = q3 - q1
    # inyectar un outlier apenas por encima del umbral 1.5*IQR
    X.loc[0, "num_medications"] = q3 + 1.5 * iqr + 0.01
    X_f, y_f = filtrar_outliers_train(X, y)
    assert len(X_f) == len(X) - 1, "el outlier apenas-fuera-del-umbral debe eliminarse"
    assert len(y_f) == len(X_f)
    # y un valor exactamente EN el umbral debe conservarse (between es inclusivo)
    X2, y2 = _train_sintetico()
    q1b, q3b = X2["num_medications"].quantile(0.25), X2["num_medications"].quantile(0.75)
    X2.loc[0, "num_medications"] = q3b + 1.5 * (q3b - q1b)
    X2_f, _ = filtrar_outliers_train(X2, y2)
    assert len(X2_f) == len(X2), "el valor exactamente en el umbral debe conservarse"


def test_filtro_devuelve_solo_features_finales():
    """Tras el filtro, las columnas auxiliares number_* deben desaparecer."""
    X, y = _train_sintetico()
    X_f, _ = filtrar_outliers_train(X, y)
    assert list(X_f.columns) == FEATURES_FINALES
    assert "number_outpatient" not in X_f.columns


def test_split_reproducible():
    """Dos llamadas al split deben producir exactamente los mismos conjuntos."""
    X, y = _train_sintetico(200)
    # y con dos clases de estadía para que stratify tenga sentido
    y = pd.Series(([3.0, 4.0] * 100)[:200], name="time_in_hospital")
    a = split_estratificado(X, y)
    b = split_estratificado(X, y)
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_frame_equal(a[1], b[1])
    pd.testing.assert_series_equal(a[2], b[2])
    pd.testing.assert_series_equal(a[3], b[3])


def test_registrar_hace_upsert_no_append(tmp_path):
    """Re-registrar el mismo modelo debe REEMPLAZAR su fila, nunca duplicarla."""
    ruta = tmp_path / "metricas.csv"
    registrar("modelo_x", {"mae": 2.0, "rmse": 3.0, "r2": 0.1}, ruta=ruta)
    registrar("modelo_x", {"mae": 1.5, "rmse": 2.5, "r2": 0.2}, ruta=ruta)
    tabla = pd.read_csv(ruta)
    assert len(tabla) == 1, "el mismo modelo no puede aparecer dos veces"
    assert tabla.loc[0, "mae"] == pytest.approx(1.5), "debe quedar el valor nuevo"
    registrar("modelo_y", {"mae": 2.2, "rmse": 3.1, "r2": 0.05}, ruta=ruta)
    tabla = pd.read_csv(ruta)
    assert len(tabla) == 2
    assert tabla["modelo"].is_unique
