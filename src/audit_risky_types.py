
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

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
    "Logistic Regression": ("logistic_regression.joblib", 0.65),
    "Random Forest": ("random_forest.joblib", 0.20),
    "XGBoost": ("xgboost.joblib", 0.07),
}


def evaluate(name, y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "model": name,
        "threshold": threshold,
        "rows": len(y_true),
        "fraud_cases": int(y_true.sum()),
        "pr_auc": average_precision_score(
            y_true, probabilities
        ),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }


def main():
    test = pd.read_csv(DATA_DIR / "test.csv")

    # Focus on transaction types where fraud appears in this dataset.
    risky = test[test["type"].isin(["CASH_OUT", "TRANSFER"])].copy()

    X = risky[FEATURES]
    y = risky["isFraud"].astype(int).to_numpy()

    print(f"Risky-type test rows: {len(risky):,}")
    print(f"Fraud cases: {int(y.sum()):,}")
    print(f"Fraud rate: {y.mean():.4%}")

    results = []

    # Simple baseline: flag every CASH_OUT and TRANSFER.
    baseline_predictions = [1] * len(risky)
    tn, fp, fn, tp = confusion_matrix(
        y, baseline_predictions, labels=[0, 1]
    ).ravel()

    results.append({
        "model": "Flag all CASH_OUT/TRANSFER",
        "threshold": None,
        "rows": len(y),
        "fraud_cases": int(y.sum()),
        "pr_auc": None,
        "precision": precision_score(
            y, baseline_predictions, zero_division=0
        ),
        "recall": recall_score(
            y, baseline_predictions, zero_division=0
        ),
        "f1": f1_score(y, baseline_predictions, zero_division=0),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    })

    for name, (filename, threshold) in MODELS.items():
        model = joblib.load(MODEL_DIR / filename)
        probabilities = model.predict_proba(X)[:, 1]

        results.append(
            evaluate(name, y, probabilities, threshold)
        )

    output = pd.DataFrame(results)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "risky_type_test_audit.csv"
    output.to_csv(path, index=False)

    print("\n=== RISKY-TYPE TEST RESULTS ===")
    print(output.round(4).to_string(index=False))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
