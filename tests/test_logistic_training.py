import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.training.logistic_baseline import (
    TrainingSettings,
    _assert_no_target_proxy_features,
    _find_target_proxy_features,
    train_one_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"


def test_train_one_dataset_creates_artifacts(tmp_path: Path) -> None:
    settings = TrainingSettings(
        random_state=42,
        max_iter=500,
        c_grid=(0.1, 1.0),
        cv_max_folds=3,
    )

    result = train_one_dataset(
        dataset_name="diabetes",
        data_dir=DATA_DIR,
        models_dir=tmp_path,
        settings=settings,
    )

    output_dir = Path(result["output_dir"])

    assert (output_dir / "logistic_pipeline.joblib").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "feature_schema.json").exists()
    assert (output_dir / "train_config.json").exists()


def test_trained_model_predicts_expected_length(tmp_path: Path) -> None:
    settings = TrainingSettings(
        random_state=42,
        max_iter=500,
        c_grid=(1.0,),
        cv_max_folds=3,
    )

    result = train_one_dataset(
        dataset_name="diabetes",
        data_dir=DATA_DIR,
        models_dir=tmp_path,
        settings=settings,
    )

    model = joblib.load(result["model_path"])

    frame = pd.read_csv(DATA_DIR / "diabetes_encoded.csv")
    sample = frame.drop(columns=["Outcome"]).head(5)

    predictions = model.predict(sample)
    probabilities = model.predict_proba(sample)

    assert len(predictions) == 5
    assert probabilities.shape[0] == 5


def test_kidney_schema_excludes_affected_with_reason(tmp_path: Path) -> None:
    settings = TrainingSettings(
        random_state=42,
        max_iter=500,
        c_grid=(0.1,),
        cv_max_folds=3,
    )

    result = train_one_dataset(
        dataset_name="kidney",
        data_dir=DATA_DIR,
        models_dir=tmp_path,
        settings=settings,
    )

    schema_path = Path(result["output_dir"]) / "feature_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert "affected" not in schema["feature_columns"]
    assert "affected" in schema["dropped_columns"]
    assert schema["dropped_column_reasons"]["affected"] == "target leakage: perfect inverse proxy for class"


def test_train_config_includes_target_leakage_guard_metadata(tmp_path: Path) -> None:
    settings = TrainingSettings(
        random_state=42,
        max_iter=500,
        c_grid=(0.1,),
        cv_max_folds=3,
    )

    result = train_one_dataset(
        dataset_name="diabetes",
        data_dir=DATA_DIR,
        models_dir=tmp_path,
        settings=settings,
    )

    config_path = Path(result["output_dir"]) / "train_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    guard = config["target_leakage_guard"]

    assert guard["enabled"] is True
    assert guard["thresholds"]["warning"]["equality_match_rate"] == 0.9
    assert guard["thresholds"]["warning"]["inverse_match_rate"] == 0.9
    assert guard["thresholds"]["hard_fail"]["equality_match_rate"] == 0.995
    assert guard["thresholds"]["hard_fail"]["inverse_match_rate"] == 0.995
    assert isinstance(guard["checked_feature_count"], int)
    assert guard["checked_feature_count"] >= 1
    assert "hard_failures" in guard
    assert "warning_flagged_features" in guard
    assert "skipped_feature_count" in guard
    assert "note" in guard


def test_leakage_guard_raises_on_inverse_proxy() -> None:
    X = pd.DataFrame(
        {
            "safe_feature": [0, 1, 2, 3, 4, 5],
            "proxy": [1, 0, 1, 0, 1, 0],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1], name="class")

    findings = _find_target_proxy_features(X, y)

    with pytest.raises(ValueError, match="Potential target leakage"):
        _assert_no_target_proxy_features(findings, dataset_name="synthetic", target_column="class")
