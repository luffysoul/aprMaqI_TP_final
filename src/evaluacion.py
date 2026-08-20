"""Evaluación estandarizada y registro acumulado de métricas.

Todas las familias de modelos se evalúan con las mismas tres métricas sobre
el MISMO test (completo, sin filtrar). El registro en resultados/metricas.csv
hace UPSERT por nombre de modelo: re-ejecutar un notebook actualiza su fila,
nunca la duplica.

Además del test, cada modelo registra `mae_cv`: su MAE de VALIDACIÓN CRUZADA
(KFold 5, seed 42, solo sobre train). Esa es la columna con la que se elige el
ganador en el notebook 07; el test se reserva para estimar sin sesgo el error
del ganador ya elegido. Para que la comparación sea homologable, `mae_cv` se
calcula con el MISMO protocolo (KFold-5 sobre el train completo) para las nueve
filas — incluidos SVR-RBF y XGBoost, cuyas BÚSQUEDAS de hiperparámetros usaron
protocolos más baratos (submuestra y holdout interno, documentados en NB04/NB06).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import DIR_RESULTADOS

RUTA_METRICAS = DIR_RESULTADOS / "metricas.csv"

# Orden canónico de columnas del registro. `mae_cv` va primero entre las
# métricas porque es el criterio de SELECCIÓN; las de test vienen después
# porque son el criterio de REPORTE.
COLUMNAS = [
    "modelo", "mae_cv", "mae", "rmse", "r2", "tiempo_s", "pred_test_s", "notas",
]


def evaluar(modelo, X_test, y_test) -> dict[str, float]:
    """MAE, RMSE y R² del modelo (ya fiteado) sobre el test.

    Incluye `pred_test_s`: el tiempo de predicción sobre el test completo —
    la latencia que pagaría el hospital por puntuar toda una cartera de
    admisiones (relevante para KNN y SVR, cuyo costo vive en la predicción).
    """
    t0 = time.perf_counter()
    pred = modelo.predict(X_test)
    pred_test_s = time.perf_counter() - t0
    return {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
        "pred_test_s": round(pred_test_s, 2),
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

    # Orden canónico: primero las columnas conocidas, después cualquier extra
    ordenadas = [c for c in COLUMNAS if c in tabla.columns]
    ordenadas += [c for c in tabla.columns if c not in ordenadas]
    tabla = tabla[ordenadas]

    tabla.to_csv(ruta, index=False)
    return tabla


def evaluar_y_registrar(
    nombre: str, modelo, X_test, y_test, notas: str = "", ruta: Path | None = None
) -> dict[str, float]:
    """Atajo: evalúa sobre test y registra con upsert. Devuelve las métricas."""
    metricas = evaluar(modelo, X_test, y_test)
    registrar(nombre, metricas, notas=notas, ruta=ruta)
    return metricas


RUTA_CACHE_CV = DIR_RESULTADOS / "cv_cache.json"


def _leer_cache() -> dict:
    if RUTA_CACHE_CV.exists():
        return json.loads(RUTA_CACHE_CV.read_text(encoding="utf-8"))
    return {}


def mae_cv_cacheado(clave: str, pipe, X, y, cv, verbose: bool = True) -> float:
    """MAE de validación cruzada, memoizado **fold por fold** en disco.

    Mismo principio que el storage sqlite de Optuna: las validaciones caras se
    pagan una vez y se reutilizan, de modo que un Run All completo no vuelva a
    pagarlas. La diferencia es la granularidad: cada fold se cachea por separado,
    así una corrida interrumpida no pierde el trabajo ya hecho.

    `clave` debe identificar unívocamente modelo + configuración + protocolo:
    si cambia cualquiera de los tres, la clave cambia y la CV se recalcula.
    """
    import time as _time

    from sklearn.base import clone
    from sklearn.metrics import mean_absolute_error

    cache = _leer_cache()
    maes = []
    for k, (idx_tr, idx_val) in enumerate(cv.split(X, y)):
        clave_fold = f"{clave}|fold{k}"
        if clave_fold in cache:
            maes.append(float(cache[clave_fold]))
            if verbose:
                print(f"  fold {k}: {cache[clave_fold]:.4f} (cacheado)")
            continue

        t0 = _time.perf_counter()
        modelo = clone(pipe)
        modelo.fit(X.iloc[idx_tr], y.iloc[idx_tr])
        mae = float(mean_absolute_error(y.iloc[idx_val], modelo.predict(X.iloc[idx_val])))
        transcurrido = _time.perf_counter() - t0

        cache = _leer_cache()          # relectura: otro proceso pudo escribir
        cache[clave_fold] = mae
        RUTA_CACHE_CV.parent.mkdir(parents=True, exist_ok=True)
        RUTA_CACHE_CV.write_text(
            json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8"
        )
        maes.append(mae)
        if verbose:
            print(f"  fold {k}: {mae:.4f} ({transcurrido:.0f} s, calculado y cacheado)")

    return float(np.mean(maes))
