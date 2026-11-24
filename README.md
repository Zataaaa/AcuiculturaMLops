# Alerta temprana de baja de oxígeno disuelto en estanques acuícolas (MLOps en AWS)

Proyecto individual de MLOps (ITESO, Carlos [Apellidos]). Predice si el **oxígeno disuelto (OD) caerá por debajo de 4.0 mg/L en las próximas 3 horas**, a partir de temperatura, pH y OD recientes, y lo despliega con un ciclo MLOps en AWS: datos en S3, entrenamiento en SageMaker, API en Lambda, monitoreo de drift con CloudWatch/SNS y reentrenamiento automático con EventBridge.

> **Estado honesto:** es un prototipo académico. El modelo supera modestamente a la regla simple "OD actual bajo" y al azar, pero no está listo para una granja real (ver *Limitaciones*).

## Datos
*IoT labelled dataset for aquaponics fish pond water quality monitoring* (Udanor et al., Universidad de Nigeria, Nsukka; artículo en *Data in Brief*, DOI del dataset 10.34740/kaggle/dsv/2681778). Sensores IoT en estanques de acuaponia (junio a octubre de 2021). **No se incluye en el repositorio**: descárgalo de Kaggle (confirma la licencia ahí) y genera el CSV horario:

```bash
python scripts/prepare_ponds.py <carpeta_con_IoTPond*.csv> data/hourly_ponds.csv
```
Se usan los estanques 1, 2, 3, 4, 6, 7 y 9 (6,710 horas limpias). El script descarta lecturas imposibles (sensor desconectado = -127 °C, OD de 40 mg/L, etc.), agrupa por hora con la mediana y conserva las horas con al menos 20 lecturas válidas. Por fallas de sensor solo quedan entre 28 % y 67 % de las horas de cada estanque.

## Planteamiento
- **Etiqueta:** 1 si el OD mínimo de las próximas 3 h es < 4.0 mg/L.
- **Alerta temprana:** el modelo solo predice cuando el OD actual es ≥ 4.5 mg/L. Si ya está bajo, la API alerta directo (`status: "oxigeno_ya_bajo"`).
- **Features (14):** temperatura, pH y OD actuales; media, desviación y diferencia de las últimas 6 h de cada uno; hora del día (seno y coseno). Solo usa el pasado; con huecos en los datos, las ventanas se calculan sobre una malla horaria completa y se descartan las filas incompletas.
- **Split temporal global 70/15/15** (2,487 / 533 / 534 filas); umbral de decisión elegido en validación para recall ≥ 0.70.

## Resultados (conjunto de prueba, split temporal)
| Modelo | Recall | Precisión | PR-AUC | ROC-AUC | % de horas con alerta |
|---|---|---|---|---|---|
| Regla "OD actual" (sin ML) | 0.87 | 0.08 | 0.308 | 0.730 | 64 % |
| Regresión logística | 0.87 | 0.09 | 0.181 | 0.767 | 55 % |
| Gradient boosting (elegido) | 0.68 | 0.23 | **0.367** | **0.815** | 17 % |

Validación dejando un estanque fuera (promedio de 7): PR-AUC 0.326 (regla), 0.346 (regresión logística), 0.365 (gradient boosting); prevalencia promedio 0.14. La mejora sobre la regla es real pero modesta. La prueba tiene pocos eventos (prevalencia 5.8 %, unos 31) y la prevalencia cambia entre validación (17.6 %) y prueba, por lo que las cifras son inestables.

## Arquitectura AWS
| Servicio | Uso |
|---|---|
| S3 | `raw/`, `processed/`, `models/`, `predictions/` (registro de cada predicción) y `code/lambda/`; versionado activo, acceso público bloqueado |
| SageMaker | Job de entrenamiento (contenedor scikit-learn, `ml.m5.large`), lanzado a mano o por EventBridge |
| Lambda | `acuicultura-api` (URL pública), `acuicultura-drift`, `acuicultura-retrain`, `acuicultura-promote` |
| CloudWatch | Métricas custom (`max_psi`, `risk_score`, `predicciones`, `recall_semanal`) y 2 alarmas |
| SNS | Tópico `acuicultura-alertas` con suscripción por correo |
| EventBridge | 3 reglas: drift cada hora, alarma de drift → reentrenar, entrenamiento terminado → promover modelo |
| IAM | Rol `acuicultura-lambda-role` de mínimo privilegio; se reutiliza `sageMakerEjecutor` para SageMaker |

Ciclo automático: cada predicción de `/predict` se guarda en `predictions/` → cada hora `acuicultura-drift` calcula el PSI de las últimas 200 → si `max_psi > 0.2` salta la alarma → EventBridge lanza `acuicultura-retrain` (job de SageMaker) → al terminar, `acuicultura-promote` copia el modelo a `models/<job>/`. La aprobación del modelo nuevo es manual (no se reemplaza solo en la API).

La API de Lambda usa el gradient boosting exportado a JSON (`scripts/export_lambda_model.py`; árboles evaluados en Python puro, sin scikit-learn, con diferencia de 1e-16 respecto a scikit-learn). La versión FastAPI (`src/acuicultura/api.py`) carga el modelo con joblib y corre en local o en Docker.

## Estructura
```
src/acuicultura/   data.py (validación) · features.py · train.py · api.py · monitoring.py
scripts/           prepare_ponds.py (datos crudos → CSV horario) · export_lambda_model.py (modelo → JSON para Lambda)
tests/             pytest: datos, features, entrenamiento, drift, API y Lambda (29 pruebas, 93 % de cobertura)
lambdas/           handler.py (api, drift, retrain, promote) · lambda_function.py (cargador desde S3)
sagemaker/         punto de entrada para script mode
.github/workflows  CI: ruff + pytest (cobertura ≥ 80 %)
```

## Uso
```bash
pip install -r requirements-dev.txt
export PYTHONPATH=src
python scripts/prepare_ponds.py <crudos> data/hourly_ponds.csv      # ver sección Datos
python -m acuicultura.train --data data/hourly_ponds.csv --out artifacts
uvicorn acuicultura.api:app --port 8000
pytest --cov=src/acuicultura
```
Endpoints: `GET /health`, `GET /model-info`, `POST /predict` (mínimo 7 lecturas horarias consecutivas con `timestamp`, `temperature_c`, `ph`, `dissolved_oxygen_mgl`; valida rangos con Pydantic, 422 si no cumplen).

Docker: entrena primero (genera `artifacts/model.joblib`, que no se versiona) y luego `docker build -t acuicultura-mlops .` (la imagen no se ha probado todavía).

Para la Lambda: `python scripts/export_lambda_model.py artifacts lambdas/model.json data/hourly_ponds.csv` y subir `handler.py` y `model.json` a `s3://<bucket>/code/lambda/`.

## Monitoreo
- **Drift:** PSI por feature contra el perfil de entrenamiento (deciles); alarma si `max_psi > 0.2`. Las variables de hora son discretas, por lo que su PSI usa las proporciones reales de cada intervalo en el entrenamiento y no 10 % fijo (con 10 % fijo daba falsos positivos).
- **Desempeño:** `recall_semanal` cuando lleguen etiquetas reales (alarma < 0.60); hoy no hay etiquetas reales, así que solo se probó con valores de ejemplo.
- Cada predicción se registra en S3 (`predictions/`) y en CloudWatch (`risk_score`).

## Limitaciones y ética
- Datos de acuaponia de laboratorio en Nigeria, no de una granja productiva; no incluyen clima, aireación ni alimentación. La meta es llevarlo a granjas de camarón de la región (Mazatlán), pero para eso se necesitan datos reales de camarón (se buscaron datasets públicos y no existe uno con series de OD reales).
- Datos con huecos y errores de sensor; la prevalencia cambia entre periodos y la prueba tiene pocos eventos.
- El umbral de 4.0 mg/L es un umbral de estrés típico para peces de estanque pendiente de respaldar con una fuente; el de camarón sería distinto.
- Riesgos: sesgo de representación, sobreconfianza (*automation bias*), intervenciones no registradas (aireadores) que contaminan las etiquetas. Mitigación: humano en el ciclo, modelo-ficha, monitoreo continuo. **No debe usarse para decisiones reales** sin validación con datos de la granja.

Nota: `lambdas/model.json` es el modelo exportado que usa la Lambda; se regenera con el script anterior después de cada entrenamiento.
