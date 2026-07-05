# Inference Pipeline Summary

## Scope
This document summarizes the current inference implementation for:
- diabetes
- heart
- kidney

Primary module:
- `src/inference/predict.py`

Primary tests:
- `tests/test_inference_pipeline.py`

## Inputs
The inference CLI accepts one of the following input sources:
- `--input-json` (single object or list of objects)
- `--input-file` (JSON object or JSON list)
- `--input-csv` (batch records from CSV)

Supported dataset selector:
- `--dataset {diabetes,heart,kidney}`

## Artifact Usage
Per dataset, inference loads artifacts from `models/<dataset>/`:
- `logistic_pipeline.joblib` (required)
- `feature_schema.json` (required)
- `metrics.json` (optional)
- `train_config.json` (optional)

The saved pipeline artifact is used directly for prediction, so preprocessing and model logic are consistent with training.

## Validation and Normalization Flow
1. Validate dataset name against configured datasets.
2. Load artifacts and required feature columns from schema.
3. Validate each record:
- required columns must exist
- extra columns are ignored with warnings
- missing values are allowed and passed through as `NaN`
- numeric/categorical numeric-like values are coerced
4. For heart categorical features, warn when values are outside known encoded categories.
5. Apply dataset-aware suspicious-range warnings.

Validation raises clear `ValueError` messages for malformed inputs.

## Prediction Flow
1. Build dataframe and reindex to schema feature order.
2. Run `pipeline.predict`.
3. If available, run `pipeline.predict_proba` and return per-class probabilities.
4. Attach metric summary from `metrics.json` when present.
5. Build deterministic JSON response for single or batch records.

## Response Fields
Per prediction result:
- `predicted_class`
- `predicted_class_label` (when label mapping exists)
- `class_probabilities` (if supported)
- `confidence_score`
- `validation_warnings`
- `model_metric_summary`

Top-level response includes:
- `dataset`
- `record_count`
- `results`
- `artifact_paths`
- `pipeline_type`

## Heart Baseline Warning
Heart predictions include a built-in warning that the current model is a baseline logistic regression with modest macro F1 and should not be used for clinical decision-making.

## Supported Commands
Single JSON file inference:
```powershell
.\.venv\Scripts\python.exe -m src.inference.predict --dataset diabetes --input-file data/external/inference_samples/diabetes_sample.json
```

Heart single JSON file inference:
```powershell
.\.venv\Scripts\python.exe -m src.inference.predict --dataset heart --input-file data/external/inference_samples/heart_sample.json
```

Kidney single JSON file inference:
```powershell
.\.venv\Scripts\python.exe -m src.inference.predict --dataset kidney --input-file data/external/inference_samples/kidney_sample.json
```

Heart batch inference from JSON list:
```powershell
.\.venv\Scripts\python.exe -m src.inference.predict --dataset heart --input-file data/external/inference_samples/heart_batch.json --output reports/heart-inference-output.json
```

Inline JSON inference:
```powershell
.\.venv\Scripts\python.exe -m src.inference.predict --dataset diabetes --input-json '{"Pregnancies":6,"Glucose":148,"BloodPressure":72,"SkinThickness":35,"Insulin":0,"BMI":33.6,"DiabetesPedigreeFunction":0.627,"Age":50}'
```

## Test Status
Current automated tests validate:
- single-record inference for all datasets
- batch list inference for heart
- missing-required-column validation failure path

Test file:
- `tests/test_inference_pipeline.py`

## Current Constraints
- This is baseline model inference, not clinical decision support.
- Class label text mapping is optional and currently dataset-map dependent.
- CSV inference is supported by CLI, but repository sample inputs are currently JSON files.
