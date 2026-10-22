
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[1]

VALIDATION_PATH = ROOT / "data" / "processed" / "validation.csv"
MODEL_PATH = ROOT / "models" / "logistic_regression.joblib"
REPORT_PATH = ROOT / "reports" / "threshold_analysis.json"

FP_COST = 5.0
THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)

NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "log_amount",
    "amount_to_origin_balance",
    "amount_to_destination_balance",
    "origin_balance_covers_amount",
]

CATEGORICAL_FEATURES = ["type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def main():
    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            "Validation CSV not found. Run python -m src.prepare_data first."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run python -m src.train_logistic first."
        )

    validation = pd.read_csv(VALIDATION_PATH)
    model = joblib.load(MODEL_PATH)

    X = validation[FEATURES]
    y = validation["isFraud"].astype("int8").to_numpy()
    amounts = validation["amount"].to_numpy()

    probabilities = model.predict_proba(X)[:, 1]
    results = []

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype("int8")

        tn, fp, fn, tp = confusion_matrix(
            y, predictions, labels=[0, 1]
        ).ravel()

        # Assumption: every missed fraud loses its full transaction amount.
        missed_fraud_loss = float(
            amounts[(y == 1) & (predictions == 0)].sum()
        )

        investigation_cost = float(fp * FP_COST)
        total_cost = investigation_cost + missed_fraud_loss

        results.append({
            "threshold": float(threshold),
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn),
            "precision": float(
                precision_score(y, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y, predictions, zero_division=0)
            ),
            "f1": float(
                f1_score(y, predictions, zero_division=0)
            ),
            "false_positive_investigation_cost": investigation_cost,
            "missed_fraud_amount_assumed_loss": missed_fraud_loss,
            "estimated_total_cost": total_cost,
        })

    best = min(results, key=lambda row: row["estimated_total_cost"])

    report = {
        "business_assumptions": {
            "false_positive_investigation_cost": FP_COST,
            "false_negative_loss": "Full transaction amount",
            "currency": "Dataset amount units; assumed USD only if source confirms USD",
        },
        "validation_rows": int(len(validation)),
        "validation_fraud_cases": int(y.sum()),
        "average_precision": float(
            average_precision_score(y, probabilities)
        ),
        "baseline_at_0_5": next(
            row for row in results if row["threshold"] == 0.5
        ),
        "best_validation_threshold": best,
        "all_threshold_results": results,
        "warning": (
            "Threshold is selected on validation data. "
            "Do not report its validation cost as independent test performance."
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n=== THRESHOLD BUSINESS-COST ANALYSIS ===")
    print(f"False-positive cost assumption: ${FP_COST:.2f}")
    print("False-negative loss assumption: full transaction amount")

    for label, row in [
        ("Baseline threshold 0.50", report["baseline_at_0_5"]),
        ("Best validation threshold", best),
    ]:
        print(f"\n{label}")
        print(f"Threshold: {row['threshold']:.2f}")
        print(f"Precision: {row['precision']:.6f}")
        print(f"Recall: {row['recall']:.6f}")
        print(f"F1: {row['f1']:.6f}")
        print(f"False positives: {row['false_positive']:,}")
        print(f"False negatives: {row['false_negative']:,}")
        print(
            "Estimated total cost: "
            f"${row['estimated_total_cost']:,.2f}"
        )

    print(f"\nReport saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
