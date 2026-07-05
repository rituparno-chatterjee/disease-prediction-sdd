import argparse
import json
from pathlib import Path
from typing import Any


def _round_if_float(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    return value


def _extract_summary(dataset_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    task_type = metrics.get("task_type")
    val = metrics.get("validation", {})
    test = metrics.get("test", {})

    summary = {
        "dataset": dataset_name,
        "task_type": task_type,
        "rows_used": metrics.get("rows_used"),
        "best_params": metrics.get("best_params", {}),
        "validation_f1": _round_if_float(val.get("f1")),
        "test_f1": _round_if_float(test.get("f1")),
        "validation_precision": _round_if_float(val.get("precision")),
        "test_precision": _round_if_float(test.get("precision")),
        "validation_recall": _round_if_float(val.get("recall")),
        "test_recall": _round_if_float(test.get("recall")),
    }

    if task_type == "binary":
        summary["validation_roc_auc"] = _round_if_float(val.get("roc_auc"))
        summary["test_roc_auc"] = _round_if_float(test.get("roc_auc"))
        summary["validation_pr_auc"] = _round_if_float(val.get("pr_auc"))
        summary["test_pr_auc"] = _round_if_float(test.get("pr_auc"))
    else:
        summary["validation_roc_auc_ovr_macro"] = _round_if_float(val.get("roc_auc_ovr_macro"))
        summary["test_roc_auc_ovr_macro"] = _round_if_float(test.get("roc_auc_ovr_macro"))

    return summary


def build_summary(models_dir: Path) -> dict[str, Any]:
    dataset_summaries = []

    for dataset_dir in sorted(models_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue

        metrics_path = dataset_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        dataset_summaries.append(_extract_summary(dataset_dir.name, metrics))

    return {"datasets": dataset_summaries}


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_markdown_report(summary: dict[str, Any]) -> str:
    datasets = summary.get("datasets", [])
    lines: list[str] = [
        "# Training Metrics Summary",
        "",
        f"Total datasets: {len(datasets)}",
        "",
    ]

    binary = [d for d in datasets if d.get("task_type") == "binary"]
    multiclass = [d for d in datasets if d.get("task_type") == "multiclass"]

    if binary:
        lines.extend(
            [
                "## Binary Datasets",
                "",
                "| dataset | rows_used | best_C | val_f1 | test_f1 | val_roc_auc | test_roc_auc | val_pr_auc | test_pr_auc |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in binary:
            best_c = row.get("best_params", {}).get("model__C")
            lines.append(
                "| "
                + " | ".join(
                    [
                        _fmt(row.get("dataset")),
                        _fmt(row.get("rows_used")),
                        _fmt(best_c),
                        _fmt(row.get("validation_f1")),
                        _fmt(row.get("test_f1")),
                        _fmt(row.get("validation_roc_auc")),
                        _fmt(row.get("test_roc_auc")),
                        _fmt(row.get("validation_pr_auc")),
                        _fmt(row.get("test_pr_auc")),
                    ]
                )
                + " |"
            )
        lines.append("")

    if multiclass:
        lines.extend(
            [
                "## Multiclass Datasets",
                "",
                "| dataset | rows_used | best_C | val_f1_macro | test_f1_macro | val_roc_auc_ovr_macro | test_roc_auc_ovr_macro |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in multiclass:
            best_c = row.get("best_params", {}).get("model__C")
            lines.append(
                "| "
                + " | ".join(
                    [
                        _fmt(row.get("dataset")),
                        _fmt(row.get("rows_used")),
                        _fmt(best_c),
                        _fmt(row.get("validation_f1")),
                        _fmt(row.get("test_f1")),
                        _fmt(row.get("validation_roc_auc_ovr_macro")),
                        _fmt(row.get("test_roc_auc_ovr_macro")),
                    ]
                )
                + " |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate dataset metrics into one report.")
    parser.add_argument("--models-dir", default="models", help="Directory containing per-dataset model outputs.")
    parser.add_argument("--output", default="reports/training-metrics-summary.json", help="Output summary JSON path.")
    parser.add_argument(
        "--markdown-output",
        default="reports/training-metrics-summary.md",
        help="Output markdown summary path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    models_dir = Path(args.models_dir)
    summary = build_summary(models_dir)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    markdown_output_path = Path(args.markdown_output)
    markdown_output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_output_path.write_text(build_markdown_report(summary), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Summary written to: {output_path}")
    print(f"Markdown summary written to: {markdown_output_path}")


if __name__ == "__main__":
    main()
