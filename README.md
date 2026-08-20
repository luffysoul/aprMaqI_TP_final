# TP Final — Aprendizaje de Máquina I (CEIA-FIUBA)

**Alumno:** Jaime Pinzón (a2629) · Cohorte 26Co2026
**Materia:** Aprendizaje de Máquina I — Especialización en Inteligencia Artificial, LSE-FIUBA-UBA

---

## Propuesta de investigación

**Pregunta:** ¿cuántos días permanecerá internado un paciente diabético, estimado al momento de su ingreso hospitalario, a partir de la información clínica y administrativa disponible en la admisión?

**Motivación de negocio.** La duración de la estadía (*length of stay*) es uno de los principales determinantes del costo hospitalario y de la disponibilidad de camas. La métrica de éxito es directamente interpretable por la gestión: **MAE en días de estadía**, comparada contra el mejor predictor constante que puede construirse sin mirar el test.

**Qué se pretende descubrir.** La predicción es el instrumento, no el objetivo. El trabajo se plantea tres preguntas *antes* de modelar y las responde en el notebook 07:

1. **¿Alcanza la información de la admisión para anticipar la estadía, o el techo está en los datos?** Hipótesis previa: el límite está en los datos.
2. **¿Qué variables de admisión gobiernan la duración: el diagnóstico (lo que el paciente *tiene*) o la intensidad de tratamiento inicial (lo que el hospital *hace*)?**
3. **¿Cuánto importa la elección de la familia de algoritmos frente a la disciplina metodológica?**

**Problema de ML:** regresión supervisada sobre `time_in_hospital` (1–14 días). Se comparan cinco familias (KNN, SVR lineal/RBF, árbol de regresión podado, Random Forest, XGBoost) contra baselines simples. El ganador se elige **por validación cruzada** (KFold-5, seed 42, solo train) y el test se abre una única vez, ya tomada la decisión.

## Resultados principales

| | |
|---|---|
| **Ganador (por validación)** | XGBoost `reg:absoluteerror` — MAE de CV **1,607** días |
| **Su error sobre el test** | MAE **1,748** días · RMSE 2,418 · R² 0,344 |
| **Piso honesto** | 2,260 días (predecir siempre 4, la mediana del train sin filtrar) |
| **Mejora** | **0,512 días por paciente (−22,7%)** |
| **Costo** | 13 s de reentrenamiento · 0,13 s para puntuar 20.354 pacientes |

Tres hallazgos que el trabajo considera más interesantes que el ranking:

- **El techo de predictibilidad es real pero fue en parte autoinfligido.** Ningún modelo supera R² ≈ 0,36. Pero el filtro de outliers heredado del trabajo de Análisis de Datos resultó, al auditarlo, una **selección de subpoblación**: en `number_outpatient` y `number_emergency` el rango intercuartílico es cero, así que sus límites colapsan a `[0, 0]` y el criterio pasa a ser "conservar solo pacientes sin visitas previas". Descarta el 34% del train y deja al 32,5% del test sin análogos. La ablación del notebook 07 mide su costo: **0,021 días de MAE y 0,018 de R²** — tanto como toda la ventaja del ganador sobre su rival más cercano.
- **Con tolerancia de ±1 día, el modelo empata con una constante** (40,8% contra 40,9%). Su ventaja aparece en ±2 días (69,5% vs 65,3%). El modelo sirve para **planificación agregada de camas**, no como pronóstico individual de fecha de alta.
- **El error no está repartido:** crece monótonamente con la duración real (0,90 días para estadías de 3 días; 7,48 para las de 14). El 10,7% de pacientes con estadías ≥9 días concentra el **30,5% del error absoluto total**.

## Fuente de datos

> Strack, B., DeShazo, J., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records*. BioMed Research International. DOI: [10.1155/2014/781670](https://doi.org/10.1155/2014/781670)
>
> Dataset: **Diabetes 130-US Hospitals for Years 1999–2008** — UCI Machine Learning Repository. DOI: [10.24432/C5230J](https://doi.org/10.24432/C5230J) · Licencia: CC BY 4.0.

El análisis exploratorio y el preprocesamiento provienen del trabajo final de Análisis de Datos (Grupo 3, mismo autor): [CEIA_Analisis_de_datos](https://github.com/masouto94/CEIA_Analisis_de_datos). Aquí se incluye el resumen y **la auditoría** de esas decisiones (notebook 01), según el criterio de la cátedra de enfocar esta materia en entrenamiento y evaluación.

## Orden de lectura

| # | Notebook | Contenido | Ejecución\* |
|---|----------|-----------|-----------|
| 01 | `01_dataset_preprocesamiento` | Propuesta y preguntas de descubrimiento, cita, preprocesamiento resumido, **auditoría del filtro heredado**, distribución del target, split, parquets congelados, checklist anti-fuga | ~1 min |
| 02 | `02_baseline` | DummyRegressor + regresión lineal; el piso honesto; por qué la media le ganó a la mediana y qué lo causó; **familias descartadas** (Poisson, cuantílica, ordinal) | ~1 min |
| 03 | `03_knn` | KNN regressor: grilla 32 configs × CV5, curva k vs MAE, regla D7 | 3,5 min |
| 04 | `04_svr` | LinearSVR (grilla extendida por D7) + SVR-RBF (Optuna en submuestra, refit completo, **CV homologable**, limitación documentada) | 9 min |
| 05 | `05_arbol` | Demostración de sobreajuste + poda por costo-complejidad (α por CV) + árbol visualizado | 2,5 min |
| 06 | `06_ensambles` | Random Forest (OOB≈CV) + XGBoost (decisión D4, **regla D7 sobre espacios continuos**, refinamiento dirigido) + feature importance ×2 métodos ×2 modelos | 6 min |
| 07 | `07_comparacion_conclusiones` | **Selección por validación**, tabla final, **análisis de error**, **bootstrap pareado**, **ablación del filtro**, conclusiones | 1 min |

\* Con las cachés presentes (ver Reproducibilidad). La primera corrida desde cero es sustancialmente más larga.

## Cómo reproducir

```bash
# 1. Instalar uv (https://docs.astral.sh/uv/)
# 2. Entorno con versiones exactas (uv.lock)
uv sync

# 3. Registrar el kernel del entorno (necesario: el kernel "python3" global
#    suele apuntar a otro intérprete, y los notebooks fallarían al importar src)
uv run python -m ipykernel install --sys-prefix --name python3

# 4. Ejecutar los notebooks en orden 01 -> 07, SIEMPRE en secuencia:
#    metricas.csv, optuna.db y las cachés no admiten escritores concurrentes.
for n in 01_dataset_preprocesamiento 02_baseline 03_knn 04_svr \
         05_arbol 06_ensambles 07_comparacion_conclusiones; do
  uv run jupyter nbconvert --to notebook --execute --inplace notebooks/$n.ipynb
done

# 5. Tests del código compartido
uv run pytest tests/ -v
```

### Qué está versionado y por qué

El repositorio incluye deliberadamente cuatro artefactos que muchos proyectos ignorarían:

| Artefacto | Qué contiene | Por qué se versiona |
|---|---|---|
| `notebooks/optuna.db` | Los cuatro estudios de Optuna (~200 KB) | Sin él, quien clona tendría que rehacer todas las búsquedas (>1 h). Con él, un Run All completo reproduce los mismos resultados en minutos. |
| `resultados/cv_cache.json` | Los MAE por fold de las CV caras (SVR-RBF, XGBoost) | La CV homologable del SVR-RBF cuesta ~10 min (5 folds × 2 min). Se cachea **fold por fold**, así una corrida interrumpida no pierde el trabajo hecho. |
| `resultados/predicciones/*.npy` | Las predicciones sobre el test de los tres mejores modelos (~400 KB) | El análisis de error, el bootstrap pareado y la ablación del NB07 las necesitan todas juntas. Sin ellas, cada re-ejecución del notebook 07 vuelve a pagar el refit del SVR-RBF (~4 min entre `fit` y `predict`). |
| `data/processed/*.parquet` | Los splits congelados, limpios y **sin codificar** | Garantizan que los siete notebooks entrenan exactamente sobre los mismos datos. |

Las claves de caché incluyen los hiperparámetros, así que **se invalidan solas** si la configuración cambia: no hay forma de reutilizar por error un resultado viejo.

### Reproducibilidad

Semilla global `SEED=42` (`src/config.py`) en split, CV, TargetEncoder (KFold interno), submuestras, Optuna, RF y XGBoost. La URL del storage de Optuna es **absoluta** (`STORAGE_OPTUNA` en `src/config.py`), de modo que no depende del directorio desde el que se invoque.

`data/processed/*.parquet` contiene los splits limpios **sin codificar**: toda transformación con estado (OneHot, TargetEncoder, escalado) vive dentro del `Pipeline` de cada modelo (`src/pipelines.py`) y se re-fitea por fold — el diseño hace estructuralmente imposible la fuga hacia el test. El test `test_target_encoder_no_usa_el_target_de_la_fila_que_codifica` verifica ese invariante con un caso extremo (una categoría por fila): sin cross-fitting la correlación con el target sería 1; con él, cae a ~0.

**Alcance de la reproducibilidad verificada.** Las columnas de métricas (`mae_cv`, `mae`, `rmse`, `r2`) de las diez filas de `resultados/metricas.csv` se reproducen de forma determinista entre corridas. Las columnas de **tiempo** (`tiempo_s`, `pred_test_s`) no: dependen de la máquina y de la carga, y varían entre ejecuciones.

## Metodología: cuatro decisiones que vale la pena señalar

1. **El ganador se elige por validación, no por test.** `metricas.csv` tiene una columna `mae_cv` calculada con el **mismo protocolo para las diez filas** (KFold-5, seed 42, train completo). Homologarla tuvo un costo real: SVR-RBF y XGBoost habían buscado hiperparámetros con protocolos más baratos (submuestra de 16k y holdout interno del 10%), así que sus `mae_cv` se **recalcularon**. El notebook 07 rankea por esa columna, declara el ganador, y recién entonces abre el test. El mismo notebook declara la única asimetría que queda: el protocolo de *cálculo* es idéntico, pero cuatro familias reportan el mínimo de su propia búsqueda sobre esa CV (*winner’s curse*) y dos reportan una CV recalculada post-hoc — un sesgo que juega en contra del ganador, no a su favor.
2. **Las diferencias se contrastan contra el ruido.** Un bootstrap pareado (10.000 remuestreos) sobre los errores absolutos del test da intervalos para cada diferencia. Las ventajas de XGBoost sobre SVR-RBF (0,021 días) y sobre Random Forest (0,050) son estadísticamente significativas — pero de una magnitud comparable al costo del filtro heredado (0,021 días, IC95 [0,017, 0,026]), que se somete **al mismo test** en vez de citarse sin intervalo. Que los dos efectos coincidan en tamaño y en intervalo es parte de la conclusión.
3. **La regla D7 se aplica, no solo se declara.** "Si el óptimo cae en un borde del espacio explorado, refinar o extender." Auditarla encontró dos incumplimientos: la grilla de `C` de LinearSVR arrancaba justo en su óptimo, y los dos parámetros de regularización de XGBoost quedaban pegados a sus límites. Ambos espacios se extendieron; el chequeo de bordes ahora se ejecuta explícitamente, también sobre los espacios continuos de Optuna.
4. **Las decisiones heredadas se auditan.** El filtro IQR portado de Análisis de Datos se documenta columna por columna, se cuantifica el desplazamiento que induce entre train y test, y se somete a una ablación completa sobre el modelo ganador.

## Checklist de criterios oficiales

| Criterio | Dónde se satisface |
|---|---|
| 1. Trabajo en grupo (2–6) | Trabajo individual, justificado ante los docentes por correo (excepción prevista en los criterios) |
| 2. Cita de la fuente de datos | NB01 §2 (DOI del paper y del dataset, licencia) + este README |
| 3. Formato ipynb / repositorio | 7 notebooks + repositorio git |
| 4. Propuesta de investigación | NB01 §1: pregunta, motivación de negocio y **tres preguntas de descubrimiento** con hipótesis previa; respondidas una por una en NB07 §6 |
| 5. Justificación de algoritmos vs. otras opciones | Celda de apertura de NB03–NB06 (qué es / por qué acá / qué exige / ventajas y desventajas), decisión D4 con experimento controlado (NB06), y **tabla de familias descartadas** con su motivo (NB02) |
| 6. Métricas claras con lectura de negocio | `src/evaluacion.py` + NB07: tabla MAE/RMSE/R²/tiempos, análisis de error por duración real, % de aciertos dentro de ±N días, e intervalos bootstrap |
| 7. Código reproducible | `uv.lock` + seeds + parquets congelados + cachés versionadas + 8 tests |
| 8. Reflexión y caminos futuros | NB07 §6: respuesta a las tres preguntas, 7 límites honestos, 7 caminos futuros ordenados por costo-beneficio, reflexión final |
| 9–10. Plazo y entrega por aula virtual | Entrega dentro del plazo, por link al repositorio |

## Estructura

```
src/           config (semilla, rutas, storage), preprocesamiento sin estado,
               pipelines (el único lugar con fit), evaluación y cachés
notebooks/     01..07 en orden de lectura + optuna.db
data/raw/      CSV original (se descarga solo si falta)
data/processed/ splits congelados sin codificar
resultados/    metricas.csv, cv_cache.json, predicciones/
tests/         8 tests de los invariantes críticos
```
