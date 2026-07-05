# AGENTS.md

## Purpose
This file defines repository-specific guidance for coding agents and contributors working on the disease prediction project.

## Project Overview
This repository contains a notebook-first preprocessing pipeline for three datasets:
- diabetes
- heart disease
- kidney disease (CKD)

Primary workflow documentation:
- SPEC.md

Primary preprocessing notebook:
- notebooks/01_eda/dataset_preprocessing.ipynb

## Repository Layout
- data/raw: source CSV files
- data/processed: generated cleaned, standardized, and encoded datasets
- notebooks: EDA and preprocessing notebooks
- src: Python source code for reusable pipeline/model logic
- tests: automated tests
- models: saved models/artifacts
- reports: outputs and analysis summaries
- app: serving or UI components (if enabled)

## Ground Rules For Agents
1. Preserve existing behavior unless the user explicitly requests a change.
2. Prefer minimal, targeted edits over broad refactors.
3. Do not delete or overwrite generated artifacts unless requested.
4. Keep changes reproducible and deterministic.
5. Keep all text and code in ASCII unless the file already requires Unicode.

## Notebook Workflow Expectations
When modifying preprocessing logic, keep this execution order aligned with SPEC.md:
1. Token standardization
2. Label encoding
3. Feature/target split
4. Missing-target cleanup
5. Feature scaling
6. Optional diagnostics

If you change notebook behavior, update SPEC.md in the same task.

## Preprocessing Conventions
- Prefer processed standardized files before raw files when encoding.
- Preserve missing values intentionally during encoding (nullable integer handling).
- Use explicit dataset-to-target mapping:
  - diabetes -> Outcome
  - heart -> num
  - kidney -> class
- For scaling:
  - scale all eligible feature columns per dataset
  - exclude ID-like columns (id, *_id)
  - preserve original column order

## Data Leakage Policy
- Do not fit scalers on full data for final modeling pipelines.
- For training pipelines, fit preprocessors on train split only and transform validation/test using fitted objects.

## Validation Checklist After Changes
1. Run affected notebook cells in order.
2. Confirm expected keys exist in memory objects:
   - splits
   - cleaned_splits
   - scaled_splits
3. Confirm output shapes are stable and consistent.
4. Confirm scaled columns have mean near 0 and std near 1.
5. Confirm heart dataset keeps id excluded from scaling.

## Dependency Notes
Main dependency files:
- requirements.txt
- requirements-dev.txt
- requirements-all.txt

When adding dependencies:
1. Add only what is necessary.
2. Keep versions compatible with existing environment.
3. Update the relevant requirements file(s).

## Testing And Quality
- Add or update tests in tests when moving logic from notebooks into src.
- Prefer small, testable utility functions over notebook-only logic for reusable steps.
- Avoid silent failures; provide clear, actionable error messages.

## Documentation Policy
Update SPEC.md whenever any of the following changes:
- pipeline step order
- preprocessing rules
- produced artifacts
- modeling assumptions
- success criteria

Keep documentation concise, factual, and synchronized with executed notebook behavior.
