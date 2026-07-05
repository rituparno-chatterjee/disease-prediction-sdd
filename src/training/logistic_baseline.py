import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.training.config import DATASET_CONFIGS, get_input_path


DATASET_FORCED_DROP_COLUMNS: dict[str, dict[str, str]] = {
    "kidney": {
        "affected": "target leakage: perfect inverse proxy for class",
    }
}

LEAKAGE_WARNING_THRESHOLD = 0.90
LEAKAGE_HARD_FAIL_THRESHOLD = 0.995


HEART_NUMERIC_CANDIDATES: tuple[str, ...] = (
    "age",
    "trestbps",
    "chol",
    "thalach",
    "thalch",
    "oldpeak",
    "ca",
)

HEART_CATEGORICAL_CANDIDATES: tuple[str, ...] = (
    "sex",
    "dataset",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "thal",
)


@dataclass(frozen=True)
class TrainingSettings:
    random_state: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    max_iter: int = 2000
    c_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    cv_max_folds: int = 5


@dataclass
class TrainingArtifacts:
    model_path: Path
    metrics_path: Path
    schema_path: Path
    config_path: Path


def _is_id_like(column_name: str) -> bool:
    normalized = column_name.strip().lower()
    return normalized == "id" or normalized.endswith("_id")


def _load_dataset_frame(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file was not found: {data_path}")
    return pd.read_csv(data_path)


def _find_target_proxy_features(
    features: pd.DataFrame,
    target: pd.Series,
    warning_threshold: float = LEAKAGE_WARNING_THRESHOLD,
    hard_fail_threshold: float = LEAKAGE_HARD_FAIL_THRESHOLD,
) -> dict[str, Any]:
    warning_features: list[dict[str, Any]] = []
    hard_fail_features: list[dict[str, Any]] = []
    skipped_features: list[dict[str, str]] = []
    checked_feature_count = 0

    for column in features.columns:
        column_numeric = pd.to_numeric(features[column], errors="coerce")
        valid_mask = column_numeric.notna() & target.notna()
        if int(valid_mask.sum()) == 0:
            skipped_features.append({"feature": column, "reason": "no_valid_numeric_rows"})
            continue

        values = column_numeric.loc[valid_mask].to_numpy(dtype=float)
        rounded_values = np.rint(values).astype(int)
        if not np.allclose(values, rounded_values, atol=1e-8):
            skipped_features.append({"feature": column, "reason": "non_integer_like_values"})
            continue

        checked_feature_count += 1
        target_values = target.loc[valid_mask].astype(int).to_numpy()
        eq_rate = float(np.mean(rounded_values == target_values))

        inv_rate: float | None = None
        target_unique = set(np.unique(target_values).tolist())
        if target_unique <= {0, 1}:
            inv_rate = float(np.mean(rounded_values == (1 - target_values)))

        finding: dict[str, Any] = {
            "feature": column,
            "equality_match_rate": eq_rate,
            "inverse_match_rate": inv_rate,
        }

        warning_hit = eq_rate >= warning_threshold or (inv_rate is not None and inv_rate >= warning_threshold)
        hard_fail_hit = eq_rate >= hard_fail_threshold or (inv_rate is not None and inv_rate >= hard_fail_threshold)

        if warning_hit:
            warning_features.append(finding)
        if hard_fail_hit:
            hard_fail_features.append(finding)

    return {
        "enabled": True,
        "note": "Guard runs after feature selection and before split/model fitting.",
        "thresholds": {
            "warning": {
                "equality_match_rate": warning_threshold,
                "inverse_match_rate": warning_threshold,
            },
            "hard_fail": {
                "equality_match_rate": hard_fail_threshold,
                "inverse_match_rate": hard_fail_threshold,
            },
        },
        "checked_feature_count": checked_feature_count,
        "checked_feature_names": sorted(features.columns.tolist()),
        "warning_flagged_features": warning_features,
        "hard_failures": hard_fail_features,
        "skipped_feature_count": len(skipped_features),
        "skipped_features": skipped_features,
    }

def _assert_no_target_proxy_features(findings: dict[str, Any], dataset_name: str, target_column: str) -> None:
    hard_failures = findings.get("hard_failures", [])
    if not hard_failures:
        return

    parts = []
    for item in hard_failures:
        inv_rate_value = item["inverse_match_rate"]
        inv_rate_text = "None" if inv_rate_value is None else f"{float(inv_rate_value):.3f}"
        parts.append(
            f"{item['feature']} (eq_rate={float(item['equality_match_rate']):.3f}, inv_rate={inv_rate_text})"
        )
    feature_summary = ", ".join(parts)
    raise ValueError(
        "Potential target leakage detected from feature-to-target proxy mapping in "
        f"dataset '{dataset_name}' for target '{target_column}': {feature_summary}."
    )


def _build_features_and_target(
    frame: pd.DataFrame,
    target_column: str,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.Series, list[str], dict[str, str]]:
    if target_column not in frame.columns:
        raise KeyError(f"Target column '{target_column}' is missing from dataset.")

    # Coerce target to numeric for robust handling of empty tokens from encoded exports.
    target = pd.to_numeric(frame[target_column], errors="coerce")
    valid_mask = target.notna()

    cleaned = frame.loc[valid_mask].copy()
    target = target.loc[valid_mask].astype(int)

    dropped_column_reasons: dict[str, str] = {}
    forced_drop_columns = DATASET_FORCED_DROP_COLUMNS.get(dataset_name, {})

    for col in cleaned.columns:
        reason: str | None = None
        if col == target_column:
            reason = "target column"
        elif _is_id_like(col):
            reason = "id-like column"

        if col in forced_drop_columns:
            reason = forced_drop_columns[col]

        if reason is not None:
            dropped_column_reasons[col] = reason

    dropped_columns = list(dropped_column_reasons.keys())
    features = cleaned.drop(columns=dropped_columns)

    if features.empty:
        raise ValueError("No feature columns remain after dropping target and ID-like columns.")

    return features, target, dropped_columns, dropped_column_reasons


def _get_task_type(target: pd.Series) -> str:
    unique_classes = target.nunique()
    return "binary" if unique_classes == 2 else "multiclass"


def _get_cv_folds(target: pd.Series, max_folds: int) -> int:
    class_counts = target.value_counts(dropna=False)
    min_class_count = int(class_counts.min()) if not class_counts.empty else 0
    folds = min(max_folds, min_class_count)
    if folds < 2:
        raise ValueError(
            "Not enough samples per class for cross-validation. "
            f"Minimum class count is {min_class_count}."
        )
    return folds


def _resolve_heart_feature_types(feature_columns: list[str]) -> tuple[list[str], list[str]]:
    numeric_cols = [col for col in feature_columns if col in HEART_NUMERIC_CANDIDATES]
    categorical_cols = [
        col
        for col in feature_columns
        if col in HEART_CATEGORICAL_CANDIDATES and col not in numeric_cols
    ]

    unassigned_cols = [
        col
        for col in feature_columns
        if col not in numeric_cols and col not in categorical_cols
    ]

    # Default any unseen heart columns to numeric so training remains robust.
    numeric_cols.extend(unassigned_cols)
    return numeric_cols, categorical_cols


def _build_pipeline(dataset_name: str, feature_columns: list[str], max_iter: int) -> Pipeline:
    if dataset_name == "heart":
        numeric_cols, categorical_cols = _resolve_heart_feature_types(feature_columns)

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_cols,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_cols,
                ),
            ]
        )

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=max_iter,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                ),
            ]
        )

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=max_iter,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _fit_model(
    dataset_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_iter: int,
    c_grid: tuple[float, ...],
    cv_max_folds: int,
) -> GridSearchCV:
    scoring = "f1" if y_train.nunique() == 2 else "f1_macro"
    cv_folds = _get_cv_folds(y_train, cv_max_folds)

    search = GridSearchCV(
        estimator=_build_pipeline(
            dataset_name=dataset_name,
            feature_columns=X_train.columns.tolist(),
            max_iter=max_iter,
        ),
        param_grid={"model__C": list(c_grid)},
        scoring=scoring,
        cv=cv_folds,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def _safe_binary_roc_auc(y_true: pd.Series, probabilities: np.ndarray) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, probabilities[:, 1]))


def _safe_binary_pr_auc(y_true: pd.Series, probabilities: np.ndarray) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(average_precision_score(y_true, probabilities[:, 1]))


def _safe_multiclass_roc_auc(y_true: pd.Series, probabilities: np.ndarray) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro"))


def _evaluate_split(
    model: Pipeline,
    X_data: pd.DataFrame,
    y_data: pd.Series,
    task_type: str,
) -> dict[str, Any]:
    predictions = model.predict(X_data)
    probabilities = model.predict_proba(X_data)

    metrics: dict[str, Any] = {
        "f1": float(f1_score(y_data, predictions, average="binary" if task_type == "binary" else "macro")),
        "precision": float(
            precision_score(
                y_data,
                predictions,
                average="binary" if task_type == "binary" else "macro",
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_data,
                predictions,
                average="binary" if task_type == "binary" else "macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(y_data, predictions).tolist(),
    }

    if task_type == "binary":
        metrics["roc_auc"] = _safe_binary_roc_auc(y_data, probabilities)
        metrics["pr_auc"] = _safe_binary_pr_auc(y_data, probabilities)
    else:
        metrics["roc_auc_ovr_macro"] = _safe_multiclass_roc_auc(y_data, probabilities)

    return metrics


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_one_dataset(
    dataset_name: str,
    data_dir: Path,
    models_dir: Path,
    settings: TrainingSettings,
) -> dict[str, Any]:
    if dataset_name not in DATASET_CONFIGS:
        raise KeyError(f"Unsupported dataset: {dataset_name}")

    config = DATASET_CONFIGS[dataset_name]
    frame = _load_dataset_frame(get_input_path(data_dir, dataset_name))
    X_all, y_all, dropped_columns, dropped_column_reasons = _build_features_and_target(
        frame,
        config.target_column,
        dataset_name,
    )

    guard_findings = _find_target_proxy_features(
        X_all,
        y_all,
        warning_threshold=LEAKAGE_WARNING_THRESHOLD,
        hard_fail_threshold=LEAKAGE_HARD_FAIL_THRESHOLD,
    )
    _assert_no_target_proxy_features(guard_findings, dataset_name, config.target_column)

    task_type = _get_task_type(y_all)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all,
        y_all,
        train_size=settings.train_ratio,
        stratify=y_all,
        random_state=settings.random_state,
    )

    relative_val_ratio = settings.val_ratio / (settings.val_ratio + settings.test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        train_size=relative_val_ratio,
        stratify=y_temp,
        random_state=settings.random_state,
    )

    search = _fit_model(
        dataset_name=dataset_name,
        X_train=X_train,
        y_train=y_train,
        max_iter=settings.max_iter,
        c_grid=settings.c_grid,
        cv_max_folds=settings.cv_max_folds,
    )

    best_pipeline: Pipeline = search.best_estimator_

    metrics = {
        "task_type": task_type,
        "best_params": search.best_params_,
        "validation": _evaluate_split(best_pipeline, X_val, y_val, task_type),
        "test": _evaluate_split(best_pipeline, X_test, y_test, task_type),
        "dataset_rows": int(len(frame)),
        "rows_used": int(len(X_all)),
    }

    output_dir = models_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = TrainingArtifacts(
        model_path=output_dir / "logistic_pipeline.joblib",
        metrics_path=output_dir / "metrics.json",
        schema_path=output_dir / "feature_schema.json",
        config_path=output_dir / "train_config.json",
    )

    joblib.dump(best_pipeline, artifacts.model_path)

    _save_json(artifacts.metrics_path, metrics)
    _save_json(
        artifacts.schema_path,
        {
            "feature_columns": X_all.columns.tolist(),
            "dropped_columns": dropped_columns,
            "dropped_column_reasons": dropped_column_reasons,
            "target_column": config.target_column,
        },
    )
    _save_json(
        artifacts.config_path,
        {
            "dataset": dataset_name,
            "input_file": config.input_file,
            "settings": {
                **asdict(settings),
                "c_grid": list(settings.c_grid),
            },
            "class_distribution": y_all.value_counts().to_dict(),
            "target_leakage_guard": guard_findings,
        },
    )

    return {
        "dataset": dataset_name,
        "output_dir": str(output_dir),
        "metrics_path": str(artifacts.metrics_path),
        "model_path": str(artifacts.model_path),
    }


def train_datasets(
    dataset_names: list[str],
    data_dir: Path,
    models_dir: Path,
    settings: TrainingSettings,
) -> dict[str, Any]:
    results = []
    for dataset_name in dataset_names:
        results.append(train_one_dataset(dataset_name, data_dir, models_dir, settings))
    return {"results": results}
