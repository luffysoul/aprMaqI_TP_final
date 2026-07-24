"""Constructor del preprocesador con estado — ÚNICO lugar del proyecto con fit.

Todos los notebooks de modelos usan exactamente el mismo preprocesador,
cambiando solo el flag de escalado. Como vive dentro del Pipeline de cada
modelo, en validación cruzada se re-fitea por fold automáticamente: el
TargetEncoder (que usa el target) jamás ve datos del fold de validación
ni del test.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from src.config import FEATURE_AGE, FEATURES_DIAG, FEATURES_NUMERICAS, SEED


def build_preprocessor(scale: bool) -> Pipeline:
    """Devuelve el preprocesador común a todos los modelos.

    - OneHot sobre `age` (10 rangos etarios; handle_unknown='ignore').
    - TargetEncoder sobre `diag_1/2/3` (categorías ICD-9; target continuo).
      Internamente usa CV propio (parámetro `cv`) para que ni siquiera el
      train se codifique con su propio target de forma directa.
    - Passthrough de las 4 numéricas.
    - StandardScaler final opcional (KNN/SVR: scale=True; árboles: False).

    El resultado transforma las 8 columnas limpias en 17 efectivas
    (4 numéricas + 10 dummies de edad + 3 diagnósticos codificados),
    igual que la matriz final del trabajo de AdD.
    """
    columnas = ColumnTransformer(
        transformers=[
            (
                "age_onehot",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                [FEATURE_AGE],
            ),
            (
                "diag_target",
                TargetEncoder(
                    target_type="continuous",
                    cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
                ),
                FEATURES_DIAG,
            ),
            ("numericas", "passthrough", FEATURES_NUMERICAS),
        ],
        verbose_feature_names_out=False,
    )
    pasos = [("columnas", columnas)]
    if scale:
        pasos.append(("escalado", StandardScaler()))
    return Pipeline(pasos)
