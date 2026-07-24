# TP Final — Aprendizaje de Máquina I (CEIA-FIUBA)

**Alumno:** Jaime Pinzón (a2629) · Cohorte 26Co2026
**Materia:** Aprendizaje de Máquina I — Especialización en Inteligencia Artificial, LSE-FIUBA-UBA

---

## Propuesta de investigación

**Pregunta:** ¿Cuántos días permanecerá internado un paciente diabético, estimado al momento de su ingreso hospitalario, a partir de la información clínica y administrativa disponible en la admisión?

**Motivación de negocio.** La duración de la estadía (*length of stay*) es uno de los principales determinantes del costo hospitalario y de la disponibilidad de camas. Un modelo que prediga los días de internación con un error medio bajo permite: (a) planificar la ocupación de camas y quirófanos, (b) dimensionar personal de enfermería por adelantado, y (c) detectar tempranamente pacientes con riesgo de estadía prolongada. La métrica de éxito es directamente interpretable por la gestión: **MAE en días de estadía**, comparada contra un piso ingenuo (predecir siempre el promedio histórico).

**Problema de ML:** regresión supervisada. Target: `time_in_hospital` (1 a 14 días). Se comparan cinco familias de modelos vistas en la materia (KNN, SVR, árbol de regresión, Random Forest, XGBoost) contra baselines simples, con optimización de hiperparámetros por validación cruzada.

## Fuente de datos

> Strack, B., DeShazo, J., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records*. BioMed Research International. DOI: [10.1155/2014/781670](https://doi.org/10.1155/2014/781670)
>
> Dataset: **Diabetes 130-US Hospitals for Years 1999–2008** — UCI Machine Learning Repository. DOI: [10.24432/C5230J](https://doi.org/10.24432/C5230J) · Licencia: CC BY 4.0.

El análisis exploratorio y el preprocesamiento provienen del trabajo final de la materia Análisis de Datos (Grupo 3, mismo autor), donde se documentó el tratamiento columna por columna: [repo CEIA_Analisis_de_datos](https://github.com/masouto94/CEIA_Analisis_de_datos). Aquí se incluye solo el resumen (criterio de la cátedra: el foco de esta materia es entrenamiento y evaluación).

## Cómo reproducir

```bash
# 1. Instalar uv (https://docs.astral.sh/uv/)
# 2. Crear el entorno e instalar dependencias exactas
uv sync
# 3. Ejecutar los notebooks en orden (01 → 07) con el kernel del entorno .venv
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_dataset_preprocesamiento.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_baseline.ipynb
# ... (repetir para el resto)
# Tests del código compartido
uv run pytest tests/ -v
```

El notebook 01 descarga el dataset crudo automáticamente si no está en `data/raw/`, y persiste los splits limpios (sin codificar) en `data/processed/*.parquet`. Todos los notebooks posteriores consumen esos parquets: la codificación (OneHot/TargetEncoder) y el escalado ocurren **exclusivamente** dentro del `Pipeline` de cada modelo (`src/pipelines.py`) para impedir fuga de datos.

## Índice de notebooks

| # | Notebook | Contenido |
|---|----------|-----------|
| 01 | `01_dataset_preprocesamiento` | Propuesta, cita, dataset, preprocesamiento resumido, split, parquets |
| 02 | `02_baseline` | DummyRegressor (media/mediana) + regresión lineal de referencia |
| 03 | `03_knn` | KNN regressor + búsqueda de hiperparámetros |
| 04 | `04_svr` | SVR lineal vs RBF |
| 05 | `05_arbol` | Árbol de regresión + poda por costo-complejidad (α) |
| 06 | `06_ensambles` | Random Forest + XGBoost + feature importance |
| 07 | `07_comparacion_conclusiones` | Tabla comparativa, conclusiones, caminos futuros |

Semilla global: `SEED = 42` (definida en `src/config.py`). Resultados acumulados en `resultados/metricas.csv` (upsert por modelo).
