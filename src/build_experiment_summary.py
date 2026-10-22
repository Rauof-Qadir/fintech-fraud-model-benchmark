
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sklearn
import xgboost


ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "experiment_summary.json"

VALIDATION_FILES = {
    "Logistic Regression": "logistic_regression_validation.json",
    "Random Forest": "random_forest_validation.json",
    "XGBoost": "xgboost_validation.json",
}


def read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    test_path = REPORTS / "test_evaluation.csv"
    benchmark_path = REPORTS / "model_benchmark.csv"

    for path in [test_path, benchmark_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required report not found: {path}")

    test_df = pd.read_csv(test_path)
    benchmark_df = pd.read_csv(benchmark_path)

    experiments = []

    for model_name, filename in VALIDATION_FILES.items():
        validation_path = REPORTS / filename

        if not validation_path.exists():
            raise FileNotFoundError(
                f"Validation report not found: {validation_path}"
            )

        validation = read_json(validation_path)

        test_rows = test_df[test_df["model"] == model_name]
        benchmark_rows = benchmark_df[
            benchmark_df["model"] == model_name
        ]

        if len(test_rows) != 1:
            raise ValueError(
                f"Expected exactly one test row for {model_name}"
            )

        if len(benchmark_rows) != 1:
            raise ValueError(
                f"Expected exactly one benchmark row for {model_name}"
            )

        test = test_rows.iloc[0].to_dict()
        benchmark = benchmark_rows.iloc[0].to_dict()

        experiments.append({
            "model": model_name,
            "training_rows": validation.get(
                "training_rows",
                validation.get("sampled_training_rows")
            ),
            "original_training_rows": validation.get(
                "original_training_rows",
                validation.get("training_rows")
            ),
            "training_fraud_cases": validation.get(
                "training_fraud_cases",
                validation.get("sampled_training_fraud_cases")
            ),
            "validation_rows": validation["validation_rows"],
            "validation_fraud_cases": validation["validation_fraud_cases"],
            "validation_threshold": validation["threshold"],
            "validation_metrics": {
                "pr_auc": validation["average_precision_pr_auc"],
                "roc_auc": validation["roc_auc"],
                "precision": validation["precision"],
                "recall": validation["recall"],
                "f1": validation["f1"],
            },
            "test_threshold": float(test["threshold"]),
            "test_metrics": {
                "pr_auc": float(test["pr_auc"]),
                "roc_auc": float(test["roc_auc"]),
                "precision": float(test["precision"]),
                "recall": float(test["recall"]),
                "f1": float(test["f1"]),
            },
            "benchmark_training_rows": int(benchmark["training_rows"]),
            "benchmark_elapsed_seconds": float(
                benchmark["elapsed_seconds"]
            ),
            "test_threshold_selection_method": "Not documented in source reports",
        })

    summary = {
        "project": "FinTech Fraud Model Benchmark",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "data_source": "Configured PaySim-style transaction dataset",
        "split_strategy": "Chronological",
        "split_boundaries": {
            "training_max_step": 520,
            "validation_max_step": 631,
            "test_steps": "632-743",
        },
        "important_limitations": [
            "Dataset is synthetic; metrics do not establish production performance.",
            "Training sizes differ across models.",
            "Test threshold selection method is not established by these reports.",
            "Validation and test distributions differ; monitor temporal drift.",
        ],
        "experiments": experiments,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(f"Experiment summary saved: {OUTPUT}")
    print(f"Models recorded: {len(experiments)}")
    print(f"Python version: {summary['environment']['python']}")


if __name__ == "__main__":
    main()
