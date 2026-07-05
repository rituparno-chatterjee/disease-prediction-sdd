from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    input_file: str
    target_column: str


DATASET_CONFIGS = {
    "diabetes": DatasetConfig(
        name="diabetes",
        input_file="diabetes_encoded.csv",
        target_column="Outcome",
    ),
    "heart": DatasetConfig(
        name="heart",
        input_file="heart_disease_uci_encoded.csv",
        target_column="num",
    ),
    "kidney": DatasetConfig(
        name="kidney",
        input_file="ckd-dataset-v2_encoded.csv",
        target_column="class",
    ),
}


def get_input_path(data_dir: Path, dataset_name: str) -> Path:
    return data_dir / DATASET_CONFIGS[dataset_name].input_file
