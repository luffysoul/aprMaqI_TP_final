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


# --- Tests del preprocesador con estado: la pieza donde una fuga sería posible ---


def test_icd9_mapea_los_bordes_de_cada_rango():
    """Tabla de casos del mapeo ICD-9: prefijos, rango de diabetes y bordes."""
    from src.preprocesamiento import icd9_a_categoria

    casos = {
        "250.83": 20, "250": 20, "250.99": 20,  # diabetes aislada
        "V27": 18, "V45.81": 18,                # códigos V
        "E909": 19, "E878.1": 19,               # códigos E
        "?": 0, None: 0, "NaN": 0,              # desconocidos
        "1": 1, "139": 1, "140": 2, "239": 2,   # bordes inferiores/superiores
        "249.9": 3, "251": 3,                   # 240-279 rodeando a diabetes
        "390": 7, "459": 7,                     # circulatorio
        "800": 17, "999": 17,                   # lesiones
        "0": 0, "1000": 0,                      # fuera de todo rango -> 0
    }
    for codigo, esperado in casos.items():
        assert icd9_a_categoria(codigo) == esperado, f"{codigo!r} -> {esperado}"


def test_preprocesador_produce_17_columnas_y_no_escala_si_no_se_pide():
    """build_preprocessor debe dar las 17 columnas efectivas del trabajo de AdD."""
    from src.pipelines import build_preprocessor

    X, y = _train_sintetico(200)
    y = pd.Series(np.linspace(1, 12, 200), name="time_in_hospital")
    X = X[FEATURES_FINALES]
    # age con las 10 franjas para que el OneHot genere las 10 dummies
    franjas = [f"[{i}-{i + 10})" for i in range(0, 100, 10)]
    X["age"] = pd.Categorical([franjas[i % 10] for i in range(200)])

    prep = build_preprocessor(scale=False)
    Xt = prep.fit_transform(X, y)
    assert Xt.shape[1] == 17, f"se esperaban 17 columnas efectivas, hay {Xt.shape[1]}"

    # Con scale=True la salida queda estandarizada (media ~0, desvío ~1)
    Xt_esc = build_preprocessor(scale=True).fit_transform(X, y)
    assert abs(Xt_esc.mean()) < 1e-8, "scale=True debe centrar las columnas"


def test_target_encoder_no_usa_el_target_de_la_fila_que_codifica():
    """Invariante anti-fuga central del TP: el TargetEncoder hace cross-fitting.

    Caso extremo y decisivo: cada fila lleva su PROPIA categoría única. Si el
    encoder codificara con el target de la fila misma, la columna resultante
    sería idéntica a `y` (correlación 1) y el modelo recibiría la respuesta
    servida. Con el CV interno de KFold(5), la categoría de una fila nunca
    aparece en los folds con los que se calcula su codificación, así que el
    encoder cae al valor por defecto (la media global) y la correlación con
    `y` se desploma. Es exactamente la fuga que `build_preprocessor` previene.
    """
    from src.pipelines import build_preprocessor

    n = 500
    rng = np.random.default_rng(0)
    y = pd.Series(rng.uniform(1, 14, size=n), name="time_in_hospital")
    categorias_unicas = np.arange(n, dtype=np.int32)

    X = pd.DataFrame({c: np.full(n, 5.0) for c in FEATURES_FINALES if c != "age"})
    for c in ["diag_1", "diag_2", "diag_3"]:
        X[c] = categorias_unicas
    X["age"] = pd.Categorical(["[50-60)"] * n)
    X = X[FEATURES_FINALES]

    prep = build_preprocessor(scale=False)
    Xt = prep.fit_transform(X, y)
    col_diag1 = Xt[:, list(prep.get_feature_names_out()).index("diag_1")]

    # Sin cross-fitting (media del target por categoría) la columna SERÍA y.
    directa = pd.Series(y.values).groupby(categorias_unicas).transform("mean").values
    assert abs(np.corrcoef(directa, y)[0, 1]) == pytest.approx(1.0), (
        "el caso de prueba debe ser tal que la codificación directa filtre y"
    )

    corr_cv = abs(np.corrcoef(col_diag1, y)[0, 1])
    assert corr_cv < 0.2, (
        "el TargetEncoder debe hacer cross-fitting: con una categoría por fila "
        f"su correlación con y debería ser ~0, y es {corr_cv:.3f}"
    )


def test_evaluar_calcula_rmse_como_raiz_del_mse():
    """RMSE debe ser sqrt(MSE), no MSE — error clásico que invalidaría la tabla."""
    from src.evaluacion import evaluar

    class ModeloFalso:
        def predict(self, X):
            return np.asarray([1.0, 2.0, 3.0, 4.0])

    y_real = pd.Series([2.0, 2.0, 5.0, 4.0])
    m = evaluar(ModeloFalso(), None, y_real)
    # errores: -1, 0, -2, 0 -> MAE = 3/4 = 0.75 ; MSE = 5/4 = 1.25
    assert m["mae"] == pytest.approx(0.75)
    assert m["rmse"] == pytest.approx(np.sqrt(1.25))
    assert m["rmse"] != pytest.approx(1.25), "rmse no puede ser el MSE sin raíz"
