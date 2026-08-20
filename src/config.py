"""Configuración global del TP: semilla y rutas."""

from pathlib import Path

SEED = 42

RAIZ = Path(__file__).resolve().parent.parent
DIR_RAW = RAIZ / "data" / "raw"
DIR_PROCESSED = RAIZ / "data" / "processed"
DIR_RESULTADOS = RAIZ / "resultados"

DIR_NOTEBOOKS = RAIZ / "notebooks"

CSV_CRUDO = DIR_RAW / "diabetic_data.csv"

# Storage sqlite de Optuna, en ruta ABSOLUTA. Los notebooks se ejecutan con el
# cwd en notebooks/ desde Jupyter pero desde la raiz con nbconvert segun como se
# invoque, y una URL relativa ("sqlite:///optuna.db") crearia dos bases distintas
# segun desde donde se corra -- perdiendo silenciosamente los trials ya hechos.
STORAGE_OPTUNA = f"sqlite:///{(DIR_NOTEBOOKS / 'optuna.db').as_posix()}"
URL_UCI = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)

TARGET = "time_in_hospital"

# Variables que sobrevivieron la selección de features del trabajo de AdD
# (get_cols en CEIA_Analisis_de_datos/trabajo_final/utils/auxiliares.py).
FEATURES_NUMERICAS = [
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_diagnoses",
]
FEATURE_AGE = "age"
FEATURES_DIAG = ["diag_1", "diag_2", "diag_3"]
FEATURES_FINALES = FEATURES_NUMERICAS + [FEATURE_AGE] + FEATURES_DIAG

# Columnas numéricas sobre las que AdD calculó el filtro IQR en train
# (X_train.describe().columns en el notebook original). Las tres "number_*"
# solo participan del filtrado de filas y luego se descartan.
COLS_FILTRO_IQR = [
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]
