
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
THRESHOLD_REPORT_PATH = REPORT_DIR / "threshold_benchmark.csv"

FP_COST = 5.0

FEATURES = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "log_amount",
    "amount_to_origin_balance",
    "amount_to_destination_balance",
    "origin_balance_covers_amount",
]

MODELS = {
    "Logistic Regression": {"file": "logistic_regression.joblib"},
    "Random Forest": {"file": "random_forest.joblib"},
    "XGBoost": {"file": "xgboost.joblib"},
}


def load_validation_thresholds():
    if not THRESHOLD_REPORT_PATH.exists():
        raise FileNotFoundError(
            "Validation threshold report missing. Run "
            "python src/compare_thresholds.py first."
        )

    table = pd.read_csv(THRESHOLD_REPORT_PATH)
    required = {"model", "best_threshold"}
    if not required.issubset(table.columns):
        raise ValueError(
            f"Threshold report must contain columns: {sorted(required)}"
        )

    if table["model"].duplicated().any():
        raise ValueError("Duplicate model names in threshold report.")

    thresholds = table.set_index("model")["best_threshold"].to_dict()
    missing = set(MODELS) - set(thresholds)
    if missing:
        raise ValueError(f"Missing validation thresholds for: {sorted(missing)}")

    for name in MODELS:
        value = float(thresholds[name])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid threshold for {name}: {value}")
        thresholds[name] = value

    return thresholds


def main():
    test = pd.read_csv(DATA_DIR / "test.csv")

    X_test = test[FEATURES]
    y_test = test["isFraud"].astype(int).to_numpy()
    amounts = test["amount"].to_numpy(dtype=float)

    print(f"Test rows: {len(test):,}")
    print(f"Actual fraud cases: {int(y_test.sum()):,}")
    print(f"Fraud rate: {y_test.mean():.4%}")

    thresholds = load_validation_thresholds()
    rows = []

    for model_name, config in MODELS.items():
        model = joblib.load(MODEL_DIR / config["file"])
        threshold = thresholds[model_name]

        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()

        fp_cost = fp * FP_COST
        missed_fraud_amount = float(amounts[
            (y_test == 1) & (predictions == 0)
        ].sum())
        estimated_cost = fp_cost + missed_fraud_amount

        row = {
            "model": model_name,
            "threshold": threshold,
            "pr_auc": average_precision_score(
                y_test, probabilities
            ),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "false_positive_cost_units": fp_cost,
            "missed_fraud_amount_units": missed_fraud_amount,
            "estimated_total_cost_units": estimated_cost,
        }

        rows.append(row)

        print(f"\n--- {model_name} ---")
        for key, value in row.items():
            if key != "model":
                print(f"{key}: {value}")

    results = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORT_DIR / "test_evaluation.csv"
    results.to_csv(output_path, index=False)

    print(f"\nSaved final test results: {output_path}")
    print("Cost figures are assumed units, not verified USD.")


if __name__ == "__main__":
    main()
