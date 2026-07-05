# Inference Plan: Logistic Baseline Serving

## Assumption
Inference should remain artifact-driven and reproducible across all three datasets (diabetes, heart, kidney), using the saved training pipelines and feature schemas already produced under `models/<dataset>/`.

## 1. Lock Inference Inputs and Contracts
- Accept dataset names: `diabetes`, `heart`, `kidney`.
- Accept input payloads as:
  - single JSON object
  - JSON list (batch)
  - CSV batch file
- Enforce required feature columns from `feature_schema.json`.
- Ignore extra columns with warnings instead of hard failure.

## 2. Keep Artifact Loading Deterministic
- Load per-dataset artifacts from `models/<dataset>/`:
  - `logistic_pipeline.joblib`
  - `feature_schema.json`
  - `metrics.json` (optional for response context)
  - `train_config.json` (optional metadata)
- Fail fast with clear errors when required artifacts are missing.

## 3. Build Strong Input Validation
- Validate required columns and record-level types.
- Normalize numeric-like values consistently.
- Preserve missing values as `NaN` so trained imputers handle them.
- Add dataset-aware warning checks for suspicious ranges.

## 4. Preserve Training-Time Feature Semantics
- Reindex inference frames to exact feature order from schema.
- Reuse full saved pipeline so preprocessing + model logic stay identical to training.
- For heart, preserve the mixed numeric/categorical preprocessing path in the pipeline.

## 5. Define Stable Response Shape
Each record response should include:
- predicted class
- optional predicted label mapping (if configured)
- class probabilities (when available)
- confidence score
- validation warnings
- model metric summary snippet (if metrics exist)

Top-level response should include dataset, record count, artifact paths, and pipeline type.

## 6. Add Safety Messaging
- Keep explicit baseline warning for heart predictions because current macro F1 is modest.
- Keep non-clinical-use disclaimer in user-facing docs.

## 7. Expand Automated Coverage
Tests in `tests/test_inference_pipeline.py` should verify:
- single-record inference for all datasets
- batch list inference
- validation failures for missing required fields
- response fields and probability presence

Potential additions:
- CSV batch path test
- missing-artifact error test
- unsupported dataset error test

## 8. Operational Outputs and Reporting
- Support `--output` to persist JSON predictions for downstream reporting.
- Keep output JSON deterministic and machine-consumable.
- Store generated inference outputs in `reports/` when needed.

## Definition of Done
- Inference runs for diabetes, heart, and kidney from saved artifacts.
- Validation catches malformed inputs with actionable messages.
- Batch and single-record flows are both supported.
- Tests pass for inference contract and key failure paths.
- Documentation reflects actual CLI behavior and limitations.
