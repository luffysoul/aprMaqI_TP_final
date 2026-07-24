"""Limpieza SIN ESTADO portada del trabajo final de Análisis de Datos (Grupo 3).

Todas las funciones de este módulo son transformaciones deterministas que no
aprenden nada de los datos (sin `fit`), con una única excepción controlada:
`filtrar_outliers_train`, cuyos umbrales IQR se calculan EXCLUSIVAMENTE sobre
el set de entrenamiento y solo filtran filas de entrenamiento. El set de test
nunca se filtra ni participa de ningún cálculo.

La codificación con estado (OneHot, TargetEncoder) NO vive acá: está en
`src.pipelines.build_preprocessor` y se fitea dentro de cada Pipeline/CV.

Referencia del preprocesamiento original (decisión por decisión):
CEIA_Analisis_de_datos/trabajo_final/trabajoFinal.ipynb
"""

from __future__ import annotations

import io
import urllib.request
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    COLS_FILTRO_IQR,
    CSV_CRUDO,
    FEATURES_DIAG,
    FEATURES_FINALES,
    SEED,
    TARGET,
    URL_UCI,
)


def descargar_si_falta() -> None:
    """Descarga y extrae el dataset de UCI solo si el CSV crudo no existe."""
    if CSV_CRUDO.exists():
        return
    CSV_CRUDO.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL_UCI) as respuesta:
        contenido = respuesta.read()
    with zipfile.ZipFile(io.BytesIO(contenido)) as zf:
        zf.extractall(CSV_CRUDO.parent)


def icd9_a_categoria(codigo) -> int:
    """Mapea un código ICD-9 a una de 21 categorías clínicas (0 a 20).

    Función determinista (sin estado) portada tal cual de AdD:
    utils/auxiliares.py::icd9_to_category. La categoría 20 aísla los códigos
    de diabetes (250.xx); 0 es "desconocido/otro" (incluye '?' y 'None').
    """
    try:
        codigo = str(codigo).strip()
        if codigo.startswith("V"):
            return 18
        if codigo.startswith("E"):
            return 19

        num = float(codigo)
        if 250 <= num <= 250.99:
            return 20
        if 1 <= num <= 139:
            return 1
        elif 140 <= num <= 239:
            return 2
        elif 240 <= num <= 279:
            return 3
        elif 280 <= num <= 289:
            return 4
        elif 290 <= num <= 319:
            return 5
        elif 320 <= num <= 389:
            return 6
        elif 390 <= num <= 459:
            return 7
        elif 460 <= num <= 519:
            return 8
        elif 520 <= num <= 579:
            return 9
        elif 580 <= num <= 629:
            return 10
        elif 630 <= num <= 679:
            return 11
        elif 680 <= num <= 709:
            return 12
        elif 710 <= num <= 739:
            return 13
        elif 740 <= num <= 759:
            return 14
        elif 760 <= num <= 779:
            return 15
        elif 780 <= num <= 799:
            return 16
        elif 800 <= num <= 999:
            return 17
        else:
            return 0
    except (ValueError, TypeError):
        return 0


def cargar_y_limpiar() -> tuple[pd.DataFrame, pd.Series]:
    """Carga el CSV crudo y aplica la limpieza sin estado de AdD.

    Devuelve (X, y) con TODAS las filas (101.766) y las columnas necesarias:
    las 8 features finales + las 3 columnas "number_*" que solo se usan para
    el filtro IQR de train. El orden de filas del CSV se preserva, de modo
    que el split con la misma semilla reproduce exactamente el de AdD.

    Pasos portados (con su celda de origen en trabajoFinal.ipynb):
    - target a float64 (celda 19).
    - Cascada de diagnósticos (celda 68): diag_2/diag_3 con '?' se imputan
      con su predecesor (diag_1 > diag_2 > diag_3). Nota de fidelidad: en el
      original, la línea que reemplazaba '?' en diag_1 no asignaba el
      resultado (no-op), por lo que diag_1 conserva '?'; es inocuo porque
      icd9_a_categoria mapea '?' a la categoría 0 de todas formas.
    - Mapeo ICD-9 → categoría (celda 137; determinista, por eso puede vivir
      acá y no en el pipeline).
    - Las imputaciones de race/payer_code/medical_specialty y el merge de
      descripciones de admisión NO se portan: ninguna de esas columnas
      sobrevive la selección de features de AdD (get_cols) y ninguna afecta
      el orden ni la cantidad de filas.
    """
    data = pd.read_csv(CSV_CRUDO)

    y = data[TARGET].astype("float64")

    diag_2 = data["diag_2"].where(data["diag_2"] != "?", other=pd.NA)
    diag_3 = data["diag_3"].where(data["diag_3"] != "?", other=pd.NA)
    diag_2 = diag_2.fillna(data["diag_1"])
    diag_3 = diag_3.fillna(diag_2)

    X = data[[c for c in dict.fromkeys(FEATURES_FINALES + COLS_FILTRO_IQR)
              if c != TARGET]].copy()
    X["diag_2"] = diag_2
    X["diag_3"] = diag_3

    for col in FEATURES_DIAG:
        X[col] = X[col].map(icd9_a_categoria).astype("int16")

    X["age"] = X["age"].astype("category")

    return X, y


def split_estratificado(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split 80/20 idéntico al de AdD: seed 42, estratificado por el target.

    El target es discreto (1 a 14 días), por lo que estratificar por su valor
    preserva la distribución de estadías en ambos conjuntos.
    """
    return train_test_split(
        X, y, test_size=0.20, random_state=SEED, shuffle=True, stratify=y
    )


def filtrar_outliers_train(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Filtro IQR de AdD: umbrales del TRAIN, aplicado SOLO al train.

    Réplica exacta de la celda 104 del original: para cada columna numérica
    de COLS_FILTRO_IQR y para el target se calculan los límites
    [Q1 - 1.5·IQR, Q3 + 1.5·IQR] sobre el train, y se conservan las filas
    donde TODAS las columnas están dentro de sus límites (el original marcaba
    outliers con NaN y hacía un único dropna, que es equivalente).

    El test NUNCA pasa por esta función: en producción no se pueden
    descartar pacientes por tener valores extremos.
    """
    mascara = pd.Series(True, index=X_train.index)
    for col in COLS_FILTRO_IQR:
        serie = X_train[col]
        q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
        iqr = q3 - q1
        inferior, superior = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mascara &= serie.between(inferior, superior)

    q1, q3 = y_train.quantile(0.25), y_train.quantile(0.75)
    iqr = q3 - q1
    mascara &= y_train.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    X_filtrado = X_train.loc[mascara, FEATURES_FINALES].reset_index(drop=True)
    y_filtrado = y_train.loc[mascara].reset_index(drop=True)
    return X_filtrado, y_filtrado


def procesar() -> dict[str, pd.DataFrame | pd.Series]:
    """Orquestador completo: crudo → limpio → split → filtro train.

    Devuelve un dict con X_train (filtrado, 8 columnas), X_test (SIN filtrar,
    8 columnas), y_train, y_test. Es la única fuente de los parquets de
    data/processed/.
    """
    descargar_si_falta()
    X, y = cargar_y_limpiar()
    X_train, X_test, y_train, y_test = split_estratificado(X, y)
    X_train, y_train = filtrar_outliers_train(X_train, y_train)
    X_test = X_test[FEATURES_FINALES].reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
