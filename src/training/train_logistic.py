import argparse
import json
from pathlib import Path

from src.training.config import DATASET_CONFIGS
from src.training.logistic_baseline import TrainingSettings, train_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train logistic regression baselines.")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", *DATASET_CONFIGS.keys()],
        help="Dataset to train. Use 'all' to train all configured datasets.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Directory containing encoded dataset CSV files.",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory where model artifacts will be saved.",
    )
    parser.add_argument("--random-state", default=42, type=int, help="Random seed for all splits.")
    parser.add_argument("--max-iter", default=2000, type=int, help="Maximum optimizer iterations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    datasets = list(DATASET_CONFIGS.keys()) if args.dataset == "all" else [args.dataset]

    settings = TrainingSettings(random_state=args.random_state, max_iter=args.max_iter)
    result = train_datasets(
        dataset_names=datasets,
        data_dir=Path(args.data_dir),
        models_dir=Path(args.models_dir),
        settings=settings,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
