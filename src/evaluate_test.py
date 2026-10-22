
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
    "Logistic Regression": {
        "file": "logistic_regression.joblib",
        "threshold": 0.65,
    },
    "Random Forest": {
        "file": "random_forest.joblib",
        "threshold": 0.20,
    },
    "XGBoost": {
        "file": "xgboost.joblib",
        "threshold": 0.07,
    },
}


def main():
    test = pd.read_csv(DATA_DIR / "test.csv")

    X_test = test[FEATURES]
    y_test = test["isFraud"].astype(int).to_numpy()
    amounts = test["amount"].to_numpy(dtype=float)

    print(f"Test rows: {len(test):,}")
    print(f"Actual fraud cases: {int(y_test.sum()):,}")
    print(f"Fraud rate: {y_test.mean():.4%}")

    rows = []

    for model_name, config in MODELS.items():
        model = joblib.load(MODEL_DIR / config["file"])
        threshold = config["threshold"]

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
