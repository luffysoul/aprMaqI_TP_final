"""Evaluación estandarizada y registro acumulado de métricas.

Todas las familias de modelos se evalúan con las mismas tres métricas sobre
el MISMO test (completo, sin filtrar). El registro en resultados/metricas.csv
hace UPSERT por nombre de modelo: re-ejecutar un notebook actualiza su fila,
nunca la duplica.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import DIR_RESULTADOS

RUTA_METRICAS = DIR_RESULTADOS / "metricas.csv"


def evaluar(modelo, X_test, y_test) -> dict[str, float]:
    """MAE, RMSE y R² del modelo (ya fiteado) sobre el test."""
    pred = modelo.predict(X_test)
    return {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
    }


def registrar(
    nombre: str,
    metricas: dict[str, float],
    notas: str = "",
    ruta: Path | None = None,
) -> pd.DataFrame:
    """UPSERT de la fila `nombre` en metricas.csv y devuelve la tabla completa.

    Si el modelo ya existe se REEMPLAZA su fila (no se agrega otra); si no,
    se agrega al final. El orden de las filas existentes se preserva.
    """
    ruta = RUTA_METRICAS if ruta is None else Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    fila = pd.DataFrame([{"modelo": nombre, **metricas, "notas": notas}])
    if ruta.exists():
        tabla = pd.read_csv(ruta)
    else:
        tabla = fila.iloc[0:0]

    if nombre in tabla["modelo"].values:
        # Reemplazo de la fila completa preservando su posición original
        posicion = int(tabla.index[tabla["modelo"] == nombre][0])
        tabla = pd.concat(
            [tabla.iloc[:posicion], fila, tabla.iloc[posicion + 1 :]],
            ignore_index=True,
        )
    else:
        tabla = pd.concat([tabla, fila], ignore_index=True)

    tabla.to_csv(ruta, index=False)
    return tabla


def evaluar_y_registrar(
    nombre: str, modelo, X_test, y_test, notas: str = "", ruta: Path | None = None
) -> dict[str, float]:
    """Atajo: evalúa sobre test y registra con upsert. Devuelve las métricas."""
    metricas = evaluar(modelo, X_test, y_test)
    registrar(nombre, metricas, notas=notas, ruta=ruta)
    return metricas
