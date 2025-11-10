# Model Card — LAS (Regresión)
**Versión:** v1  
**Entorno:** Python 3.12.7 | scikit-learn 1.4.2

## Datos
Archivo: `credito_cleandata_4model.csv`  
Shape: (60, 7)  
Objetivo (continuo): `Line credito`

## Entrenamiento
Split 80/20 (random_state=42).
Preprocesamiento: StandardScaler(num) + OHE(nominal) + OrdinalEncoder(ordinal).

## Métricas TEST
MAE=100.873 | RMSE=120.524 | R2=0.995 | EVS=0.996

## Artefactos
- `pipeline_LAS.joblib`
- `input_schema.json`
- `decision_policy.json`
