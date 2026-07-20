# Disease Prediction SDD

Notebook-first data preparation, baseline training, and inference utilities for three datasets:
- diabetes
- heart disease
- kidney disease (CKD)

## Project Goals
- Build a reproducible preprocessing workflow across all datasets.
- Train baseline logistic regression models with saved artifacts.
- Run validated single-record and batch inference from CLI.

## Repository Structure
- `data/raw`: source CSV files.
- `data/processed`: cleaned, standardized, and encoded datasets.
- `notebooks/01_eda/dataset_preprocessing.ipynb`: primary preprocessing notebook.
- `src/training`: baseline training scripts.
- `src/inference`: model loading and prediction CLI.
- `models/<dataset>`: saved model pipeline, schema, metrics, and config.
- `reports`: generated metric summaries.
- `app/ui/preview.html`: lightweight HTML preview for training and inference outputs.
- `tests`: training and inference tests.

## Setup
Use Python 3.11+ and a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you need development dependencies:

```powershell
pip install -r requirements-dev.txt
```

## Data Preprocessing Workflow
Primary workflow documentation is in `SPEC.md` and implemented in:
- `notebooks/01_eda/dataset_preprocessing.ipynb`

Related workflow docs:
- `TRAINING-PLAN.md`
- `TRAINING-SUMMARY.md`
- `INFERENCE-PLAN.md`
- `INFERENCE-SUMMARY.md`

Expected high-level order:
1. Token standardization
2. Label encoding
3. Feature/target split
4. Missing-target cleanup
5. Feature scaling
6. Optional diagnostics

Generated dataset artifacts are written to `data/processed`.

## Train Baseline Logistic Models
Train for all configured datasets:

```powershell
.\.venv\Scripts\python.exe -m src.training.train_logistic --dataset all
```

Train for a single dataset (`diabetes`, `heart`, or `kidney`):

```powershell
.\.venv\Scripts\python.exe -m src.training.train_logistic --dataset heart
```

The training command writes artifacts under:
- `models/diabetes`
- `models/heart`
- `models/kidney`

Each dataset folder includes:
- `logistic_pipeline.joblib`
- `feature_schema.json`
- `metrics.json`
- `train_config.json`

## Summarize Training Metrics
Generate JSON and Markdown summaries from saved model metrics:

```powershell
.\.venv\Scripts\python.exe -m src.training.summarize_metrics `
	--models-dir models `
	--output reports/training-metrics-summary.json `
	--markdown-output reports/training-metrics-summary.md
```

## Run Inference
Use dataset-specific model artifacts with JSON or CSV input.

Single JSON record (from sample file):

```powershell
.\.venv\Scripts\python.exe -m src.inference.predict `
	--dataset diabetes `
	--input-file data/external/inference_samples/diabetes_sample.json
```

From JSON file:

```powershell
.\.venv\Scripts\python.exe -m src.inference.predict `
	--dataset heart `
	--input-file data/external/inference_samples/heart_sample.json
```

From JSON file (batch list):

```powershell
.\.venv\Scripts\python.exe -m src.inference.predict `
	--dataset heart `
	--input-file data/external/inference_samples/heart_batch.json `
	--output reports/heart-predictions.json
```

From CSV file (batch):

```powershell
.\.venv\Scripts\python.exe -m src.inference.predict `
	--dataset heart `
	--input-csv path/to/input.csv `
	--output reports/heart-predictions.json
```

## Lightweight Preview UI
Open the static preview page to visualize saved training metrics and sample inference outputs:
- `app/ui/preview.html`

The preview renders:
- dataset-level training metrics
- human-readable predicted class labels for diabetes, heart, and kidney outputs
- encoded class IDs for traceability
- top class probability and per-class probability bars

## Run Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Notes
- This project currently provides baseline models and is not for clinical decision-making.
- Keep preprocessing behavior and documentation synchronized when changing notebook logic.
