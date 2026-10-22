
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"

REPORT_FILES = {
    "Logistic Regression": "logistic_regression_validation.json",
    "Random Forest": "random_forest_validation.json",
    "XGBoost": "xgboost_validation.json",
}

METRICS = [
    "average_precision_pr_auc",
    "roc_auc",
    "precision",
    "recall",
    "f1",
    "elapsed_seconds",
]


def main():
    rows = []

    for model_name, filename in REPORT_FILES.items():
        report_path = REPORT_DIR / filename

        if not report_path.exists():
            raise FileNotFoundError(
                f"Missing report: {report_path}"
            )

        with report_path.open("r", encoding="utf-8") as file:
            report = json.load(file)

        row = {
            "model": model_name,
            "training_rows": report.get(
                "sampled_training_rows",
                report.get("training_rows", report.get(
                    "original_training_rows"
                )),
            ),
        }

        for metric in METRICS:
            row[metric] = report.get(metric)

        matrix = report.get("confusion_matrix", {})
        row["false_positives"] = matrix.get("false_positive")
        row["false_negatives"] = matrix.get("false_negative")
        row["true_positives"] = matrix.get("true_positive")

        rows.append(row)

    df = pd.DataFrame(rows)

    output_path = REPORT_DIR / "model_benchmark.csv"
    df.to_csv(output_path, index=False)

    display_df = df.copy()

    for column in [
        "average_precision_pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
    ]:
        display_df[column] = display_df[column].map(
            lambda value: (
                f"{value:.4f}" if pd.notna(value) else "N/A"
            )
        )

    print("\n=== MODEL BENCHMARK: VALIDATION ===")
    print(display_df.to_string(index=False))
    print(f"\nCSV report saved: {output_path}")


if __name__ == "__main__":
    main()
