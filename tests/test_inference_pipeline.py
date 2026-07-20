import json
from pathlib import Path

import pytest

from src.inference.predict import run_prediction


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
SAMPLES_DIR = ROOT / "data" / "external" / "inference_samples"


def _load_sample(name: str) -> dict:
    path = SAMPLES_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "dataset_name,sample_file",
    [
        ("diabetes", "diabetes_sample.json"),
        ("heart", "heart_sample.json"),
        ("kidney", "kidney_sample.json"),
    ],
)
def test_run_prediction_single_record(dataset_name: str, sample_file: str) -> None:
    payload = _load_sample(sample_file)

    result = run_prediction(dataset_name=dataset_name, input_data=payload, models_dir=MODELS_DIR)

    assert result["dataset"] == dataset_name
    assert result["record_count"] == 1

    prediction = result["results"][0]
    assert "predicted_class" in prediction
    assert "validation_warnings" in prediction
    assert "model_metric_summary" in prediction

    assert prediction["class_probabilities"] is not None
    assert isinstance(prediction["top_class_probability"], float)

    if dataset_name == "heart":
        assert "model_warning" in prediction
        assert "baseline logistic regression model" in prediction["model_warning"]
        assert len(prediction["class_probabilities"]) >= 3


def test_run_prediction_batch_json_records() -> None:
    payload = _load_sample("heart_batch.json")

    result = run_prediction(dataset_name="heart", input_data=payload, models_dir=MODELS_DIR)

    assert result["record_count"] == 2
    assert len(result["results"]) == 2


def test_missing_required_column_raises_validation_error() -> None:
    payload = {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
    }

    with pytest.raises(ValueError, match="Missing required columns"):
        run_prediction(dataset_name="diabetes", input_data=payload, models_dir=MODELS_DIR)
