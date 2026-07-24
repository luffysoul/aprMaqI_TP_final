# TP Final — Aprendizaje de Máquina I (CEIA-FIUBA)

**Alumno:** Jaime Pinzón (a2629) · Cohorte 26Co2026
**Materia:** Aprendizaje de Máquina I — Especialización en Inteligencia Artificial, LSE-FIUBA-UBA

---

## Propuesta de investigación

**Pregunta:** ¿Cuántos días permanecerá internado un paciente diabético, estimado al momento de su ingreso hospitalario, a partir de la información clínica y administrativa disponible en la admisión?

**Motivación de negocio.** La duración de la estadía (*length of stay*) es uno de los principales determinantes del costo hospitalario y de la disponibilidad de camas. La métrica de éxito es directamente interpretable por la gestión: **MAE en días de estadía**, comparada contra el piso ingenuo de predecir siempre el promedio histórico.

**Problema de ML:** regresión supervisada sobre `time_in_hospital` (1–14 días). Se comparan cinco familias (KNN, SVR lineal/RBF, árbol de regresión podado, Random Forest, XGBoost) contra baselines simples, con hiperparámetros optimizados por validación cruzada (KFold 5, seed 42, solo sobre train) y una única evaluación final sobre el test completo.

**Resultado principal:** XGBoost (`reg:absoluteerror`) alcanza **MAE 1,746 días** vs. 2,280 del baseline (−23,4%), con 12 s de entrenamiento y 0,1 s de predicción sobre 20.354 pacientes. Ningún modelo supera R² ≈ 0,35: el techo de predictibilidad con variables de admisión es un hallazgo central del trabajo (ver notebook 07).

## Fuente de datos

> Strack, B., DeShazo, J., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records*. BioMed Research International. DOI: [10.1155/2014/781670](https://doi.org/10.1155/2014/781670)
>
> Dataset: **Diabetes 130-US Hospitals for Years 1999–2008** — UCI Machine Learning Repository. DOI: [10.24432/C5230J](https://doi.org/10.24432/C5230J) · Licencia: CC BY 4.0.

El análisis exploratorio y el preprocesamiento provienen del trabajo final de Análisis de Datos (Grupo 3, mismo autor): [CEIA_Analisis_de_datos](https://github.com/masouto94/CEIA_Analisis_de_datos). Aquí se incluye solo el resumen (notebook 01), según el criterio de la cátedra de enfocar esta materia en entrenamiento y evaluación.

## Orden de lectura y tiempos de ejecución

| # | Notebook | Contenido | Ejecución* |
|---|----------|-----------|-----------|
| 01 | `01_dataset_preprocesamiento` | Propuesta, cita, preprocesamiento resumido, split, parquets congelados, checklist anti-fuga | 26 s |
| 02 | `02_baseline` | DummyRegressor (media/mediana) + regresión lineal; evidencia media-vs-mediana | 8 s |
| 03 | `03_knn` | KNN regressor: grilla 32 configs × CV5, curva k vs MAE | 3,4 min |
| 04 | `04_svr` | LinearSVR (train completo) + SVR-RBF (Optuna en submuestra, refit completo, limitación documentada) | 9,5 min |
| 05 | `05_arbol` | Demostración de sobreajuste + poda por costo-complejidad (α por CV) + árbol visualizado | 2,5 min |
| 06 | `06_ensambles` | Random Forest (OOB≈CV) + XGBoost (decisión D4 con evidencia) + feature importance ×2 métodos ×2 modelos | 1,6 min |
| 07 | `07_comparacion_conclusiones` | Tabla final, lectura global, conclusiones, límites y caminos futuros | 8 s |

\* Tiempos de la corrida limpia de verificación (laptop 8 núcleos, con `optuna.db` presente — ver Reproducibilidad).

## Cómo reproducir

```bash
# 1. Instalar uv (https://docs.astral.sh/uv/)
# 2. Entorno con versiones exactas (uv.lock)
uv sync
# 3. Ejecutar los notebooks en orden 01 -> 07:
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_dataset_preprocesamiento.ipynb
# ... (repetir para 02..07; SIEMPRE en secuencia, nunca en paralelo:
#      metricas.csv y optuna.db no admiten escritores concurrentes)
# 4. Tests del código compartido
uv run pytest tests/ -v
```

### Dos niveles de reproducibilidad (verificados)

- **Nivel 1 — con `optuna.db` presente (ejecutado y verificado):** entorno recreado desde cero (`rm .venv && uv sync`) + Run All 01→07. Las búsquedas de Optuna son **idempotentes** (cargan los trials existentes del storage sqlite y no agregan nuevos), las grillas de KNN/LinearSVR y toda la CV se recomputan completas. Tiempo total: **~18 minutos**. Resultado verificado: las 9 filas de `resultados/metricas.csv` se reproducen **idénticas a 12 decimales**.
- **Nivel 2 — desde cero absoluto (documentado, no requerido):** borrar `optuna.db` re-ejecuta las búsquedas completas (~1,5 h). Como todos los samplers usan `TPESampler(seed=42)` y los objetivos son deterministas (CV con seed, modelos con seed), los trials se regeneran con las mismas propuestas y el resultado converge a las mismas configuraciones.

Notas de reproducibilidad: semilla global `SEED=42` (`src/config.py`) en split, CV, TargetEncoder (KFold interno), submuestras, Optuna, RF y XGBoost. `data/processed/*.parquet` contiene los splits **limpios sin codificar**: toda transformación con estado (OneHot, TargetEncoder, escalado) vive dentro del `Pipeline` de cada modelo (`src/pipelines.py`) y se re-fitea por fold — el diseño hace estructuralmente imposible la fuga de datos hacia el test.

## Checklist de criterios oficiales

| Criterio | Dónde se satisface |
|---|---|
| 1. Propuesta de investigación | NB01 §1 (celda 1) + este README |
| 2. Cita de la fuente de datos | NB01 §2 (celda 2, con DOI y licencia) + este README |
| 3. Dataset con preprocesamiento resumido | NB01 §3 (celda 4: tabla-síntesis de decisiones con link al trabajo de AdD) |
| 4. Baseline simple (sin regresión logística) | NB02 completo (celda 1 justifica por qué la logística no aplica a regresión) |
| 5. Justificación de cada algoritmo vs. otras opciones | Celda de apertura de NB03, NB04, NB05 y NB06 |
| 6. Métricas claras con lectura de negocio | `src/evaluacion.py` + NB07 (tabla con MAE/RMSE/R²/tiempos + lectura global en días de estadía) |
| 7. Código reproducible | `uv.lock` + seeds + parquets congelados + corrida limpia verificada (sección anterior) + `tests/` |
| 8. Reflexión y caminos futuros | NB07 §Conclusiones (celda final: respuesta, límites honestos, 6 caminos futuros, reflexión) |

Grupo individual: justificado ante los docentes por correo (excepción prevista en los criterios).
