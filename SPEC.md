# Disease Prediction Data Preparation Specification

### Execution Summary
Run order in notebooks/01_eda/dataset_preprocessing.ipynb:
1. Token standardization cell -> writes data/processed/*_token_standardized.csv
2. Label encoding cell -> writes data/processed/*_encoded.csv
3. Split cell -> creates splits for diabetes, heart, kidney
4. Missing-target cleanup cell -> creates cleaned_splits
5. Feature scaling cell -> creates scaled_splits for diabetes, heart, kidney
6. Optional kidney diagnostics cell -> confirms dropped target-missing rows

Use cleaned_splits for training data preparation and scaled_splits for scale-sensitive models.

## 1. Objective
Build a reproducible preprocessing workflow for three disease prediction datasets:
- Diabetes
- Heart disease
- Kidney disease (CKD)

The workflow standardizes raw data, handles missing values, removes obvious bad rows for training targets, encodes categorical variables, and prepares feature/target splits for modeling.
It also applies feature scaling across all datasets with ID-like column exclusion.
This specification ensures consistent, reproducible preprocessing for multi-disease ML models.

## 2. Scope
This specification covers the steps implemented in the notebook:
- notebooks/01_eda/dataset_preprocessing.ipynb

It also lists generated files under:
- data/processed

## 3. Input Datasets
Raw sources in data/raw:
- ckd-dataset-v2.csv
- diabetes.csv
- heart_disease_uci.csv

## 4. Environment and Dependencies
Required Python libraries include:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost

Pinned dependency files:
- requirements.txt
- requirements-dev.txt
- requirements-all.txt

## 5. Implemented Workflow

### Step 1: Initial dataset loading and summary
Implemented in the EDA notebook to inspect:
- Head rows
- Shape
- Column names
- Info and descriptive statistics
- Missing value counts and percentages

### Step 2: Missing value handling
Rule-based imputation was applied:
- Numeric columns: fill with column mean
- Categorical columns: fill with column mode

Intermediate cleaned files are written as:
- data/processed/<dataset>_cleaned.csv

### Step 3: Duplicate handling
Duplicate rows are detected and removed using pandas drop_duplicates().

### Step 4: Outlier handling (IQR)
Outliers are identified on numeric columns with IQR bounds:
- Lower = Q1 - 1.5 * IQR
- Upper = Q3 + 1.5 * IQR

Rows outside bounds on any numeric column are removed in the outlier step.

### Step 5: Token standardization (categorical cleanup)
Before encoding, noisy categorical tokens are standardized.

Missing-like/noisy tokens mapped to NA:
- empty string
- ?
- na
- n/a
- none
- null
- nan
- discrete
- class
- p
- meta

Outputs written as:
- data/processed/<dataset>_token_standardized.csv

### Step 6: Label encoding
Categorical columns are label-encoded using sklearn LabelEncoder.

Important implementation details:
- Standardized files are preferred as inputs
- Raw files are fallback only when standardized files are absent
- Missing values are preserved during encoding using nullable Int64 columns

Encoded outputs:
- data/processed/ckd-dataset-v2_encoded.csv
- data/processed/diabetes_encoded.csv
- data/processed/heart_disease_uci_encoded.csv

### Step 7: Feature and target split
Each encoded dataset is split into X and y using explicit target mapping:
- diabetes -> Outcome
- heart -> num
- kidney -> class

In-memory structure:
- splits[dataset_name]['X']
- splits[dataset_name]['y']

### Step 8: Missing target cleanup
Rows with missing target values are removed from each split.

In-memory cleaned structure:
- cleaned_splits[dataset_name]['X']
- cleaned_splits[dataset_name]['y']

Observed result from execution:
- diabetes: 0 rows dropped
- heart: 0 rows dropped
- kidney: 2 rows dropped (empty/invalid target rows)

### Step 9: Kidney dropped-row diagnostics
A diagnostic cell identifies and prints kidney rows dropped due to missing target.

Observed diagnostic result:
- Dropped kidney indices: [0, 1]
- Rows were effectively all-NaN records

### Step 10: Feature scaling for all datasets
Feature scaling is applied with sklearn StandardScaler in a loop over all available datasets.

Implementation details:
- Source preference: cleaned_splits is used when available; otherwise splits is used
- Scaling scope: all non-ID columns are scaled per dataset
- ID-like exclusion rules:
	- column name equals id
	- column name ends with _id
- Dataframe reconstruction preserves original column order while avoiding dtype-assignment conflicts

In-memory outputs per dataset:
- scaled_splits[dataset_name]['X'] -> scaled feature dataframe
- scaled_splits[dataset_name]['y'] -> target series (unchanged)
- scaled_splits[dataset_name]['scaler'] -> fitted scaler object (or None if no columns eligible)
- scaled_splits[dataset_name]['scaled_columns'] -> scaled columns list
- scaled_splits[dataset_name]['excluded_columns'] -> excluded ID-like columns

Observed result from execution:
- diabetes scaled successfully
- heart scaled successfully (id excluded)
- kidney scaled successfully
- scaled column means are approximately 0 and standard deviations are approximately 1

## 6. Artifacts Produced

### Notebook
- notebooks/01_eda/dataset_preprocessing.ipynb

### Processed Data Files
- data/processed/ckd-dataset-v2_cleaned.csv
- data/processed/diabetes_cleaned.csv
- data/processed/heart_disease_uci_cleaned.csv
- data/processed/ckd-dataset-v2_token_standardized.csv
- data/processed/diabetes_token_standardized.csv
- data/processed/heart_disease_uci_token_standardized.csv
- data/processed/ckd-dataset-v2_encoded.csv
- data/processed/diabetes_encoded.csv
- data/processed/heart_disease_uci_encoded.csv

## 7. Current Assumptions
- Encoded datasets are the primary inputs for training.
- cleaned_splits should be used for model training to avoid missing targets.
- scaled_splits should be used for models that require standardized feature scales.
- Label encoding is acceptable for current baseline modeling; one-hot encoding may be preferred for some model families.

## 8. Recommended Next Steps
1. Add stratified train/validation/test splitting per dataset.
2. Fit scalers on training splits only, then transform validation/test splits to prevent leakage.
3. Persist label encoders per dataset/column for inference-time consistency.
4. Persist fitted scalers (or preprocessing pipelines) per dataset for inference-time consistency.
5. Add baseline model training and evaluation notebooks/scripts for all three diseases.
6. Add data validation checks (schema, allowed values, missing thresholds).

## 9. Quick Execution Checklist
Use this checklist when rerunning the pipeline in the notebook.

1. Open and run imports in notebooks/01_eda/dataset_preprocessing.ipynb.
2. Run token standardization cell.
	- Expected output files: data/processed/*_token_standardized.csv
3. Run label encoding cell.
	- Expected output files: data/processed/*_encoded.csv
4. Run split cell for all datasets.
	- Expected in-memory object: splits
5. Run missing-target cleanup cell.
	- Expected in-memory object: cleaned_splits
6. Run feature scaling cell.
	- Expected in-memory object: scaled_splits
7. Optional: run kidney dropped-row diagnostics cell.
8. Use cleaned_splits and/or scaled_splits depending on model requirements.

## 10. Success Criteria
- All three encoded files exist under data/processed.
- splits contains keys: diabetes, heart, kidney.
- cleaned_splits contains keys: diabetes, heart, kidney.
- scaled_splits contains keys: diabetes, heart, kidney.
- Kidney cleanup drops only clearly invalid rows (currently 2 all-NaN target rows).
- Heart dataset excludes id from scaling while preserving column order.

## 11. Logistic Regression Baseline Training
Implementation is provided in:
- src/training/logistic_baseline.py
- src/training/train_logistic.py

Training workflow:
1. Load encoded inputs from data/processed.
2. Build features and target using explicit mapping:
	- diabetes -> Outcome
	- heart -> num
	- kidney -> class
3. Drop rows with missing targets.
4. Exclude ID-like feature columns before training:
	- column name equals id
	- column name ends with _id
5. Apply dataset-specific exclusions before training:
	- kidney -> drop affected (target leakage: perfect inverse proxy for class)
6. Run a fail-fast leakage guard that blocks training when any integer-like feature is a near-perfect direct or inverse proxy of the target.
7. Create stratified train/validation/test splits (70/15/15).
8. Train a leakage-safe scikit-learn Pipeline:
	- Diabetes and kidney use the existing numeric-only path:
		- SimpleImputer(strategy=median)
		- StandardScaler
		- LogisticRegression (class_weight=balanced)
	- Heart uses a type-aware ColumnTransformer path because it includes nominal medical categorical variables (cp, restecg, slope, thal, sex, fbs, exang, dataset):
		- Numeric branch: SimpleImputer(strategy=median) + StandardScaler
		- Categorical branch: SimpleImputer(strategy=most_frequent) + OneHotEncoder(handle_unknown="ignore")
		- LogisticRegression (class_weight=balanced)
9. Tune C with GridSearchCV and refit best pipeline.
10. Evaluate on validation and test splits.

Metrics:
- Binary tasks: F1, precision, recall, ROC-AUC, PR-AUC, confusion matrix
- Multiclass tasks: macro F1, macro precision, macro recall, ROC-AUC (OvR macro), confusion matrix

Current heart multiclass snapshot:
- Best C: 0.1
- Validation macro F1: 0.4031
- Test macro F1: 0.3712
- Validation macro ROC-AUC (OvR): 0.7882
- Test macro ROC-AUC (OvR): 0.7874

Run command:
- python -m src.training.train_logistic --dataset all

## 12. Training Artifacts Produced
For each dataset in models/<dataset>/:
- logistic_pipeline.joblib
- metrics.json
- feature_schema.json
- train_config.json

feature_schema.json contains:
- feature_columns
- dropped_columns
- dropped_column_reasons
- target_column

Cross-dataset reports:
- reports/training-metrics-summary.json
- reports/training-metrics-summary.md
