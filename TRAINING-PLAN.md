# Training Plan: Logistic Regression Baseline

## Assumption
Apply this as a baseline across all three prepared datasets (diabetes, heart, kidney) using the preprocessing contract in `SPEC.md`.
If only one dataset is needed first, run the same plan for that dataset only.

## 1. Lock Training Inputs and Targets
- Use encoded data and target mapping already defined in `SPEC.md`:
  - diabetes -> Outcome
  - heart -> num
  - kidney -> class
- Start from cleaned splits (no missing targets).
- Keep scaling leakage-safe by fitting transforms only on train data.

## 2. Create a Reusable Training Entrypoint
- Implement a training module in `src/training` that:
  - loads each dataset from `data/processed`
  - applies dataset-specific feature/target selection
  - excludes ID-like columns from scaling (especially heart `id`), consistent with `SPEC.md` and `AGENTS.md`

## 3. Build Leakage-Safe Train/Validation/Test Splits
- Use stratified split with fixed random state for reproducibility.
- Suggested split: 70/15/15 (or 80/20 with cross-validation if preferred).
- Keep class distribution checks and fail early if a class has too few samples.

## 4. Train Logistic Regression in a Pipeline
- Use a scikit-learn `Pipeline` with `StandardScaler` + `LogisticRegression`.
- Start with robust baseline settings:
  - `max_iter` large enough to converge (for example 1000-3000)
  - `class_weight=balanced` for imbalanced targets
  - solver supporting binary and multiclass
- Run a small hyperparameter search on `C` (for example log-scale grid).

## 5. Evaluate Per Dataset with the Right Metrics
- Binary tasks: ROC-AUC, PR-AUC, F1, precision, recall, confusion matrix.
- Multiclass tasks (heart, if multiclass labels remain): macro F1, macro precision/recall, one-vs-rest ROC-AUC where applicable.
- Record both validation and test results.

## 6. Save Reproducible Artifacts
For each dataset, write under `models/<dataset>/`:
- trained logistic model artifact
- scaler/pipeline artifact
- metrics report (`.json`)
- feature schema (column order and dropped columns)
- train config metadata (random seed, split ratios, hyperparameters)

## 7. Add Minimal Automated Tests
Add tests in `tests/` to verify:
- training runs end-to-end without errors
- artifacts are created for each dataset
- model can load and predict with expected feature shape
- no leakage pattern (scaler fit only on training split)

## 8. Update Project Documentation
- Update `SPEC.md` with the logistic-regression training step, artifact list, and evaluation outputs.
- Keep notebook/spec behavior synchronized with `AGENTS.md` rules.

## Definition of Done
- Logistic regression trains successfully for selected dataset(s).
- Metrics and confusion matrix are generated and stored.
- Artifacts are saved under `models` by dataset.
- Tests pass for training flow and artifact loading.
- `SPEC.md` reflects the finalized training workflow.
