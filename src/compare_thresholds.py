
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

FP_COST = 5.0
BASELINE_THRESHOLD = 0.50
THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)

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
    "Logistic Regression": "logistic_regression.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
}


def evaluate_threshold(y_true, probabilities, amounts, threshold):
    predictions = (probabilities >= threshold).astype(int)

    false_positive = (y_true == 0) & (predictions == 1)
    false_negative = (y_true == 1) & (predictions == 0)

    fp = int(false_positive.sum())
    fn = int(false_negative.sum())
    tp = int(((y_true == 1) & (predictions == 1)).sum())

    fp_cost = fp * FP_COST
    missed_fraud_amount = float(amounts[false_negative].sum())
    total_cost = fp_cost + missed_fraud_amount

    return {
        "threshold": float(threshold),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "false_positive_cost_units": fp_cost,
        "missed_fraud_amount_units": missed_fraud_amount,
        "estimated_total_cost_units": total_cost,
    }


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    validation = pd.read_csv(DATA_DIR / "validation.csv")

    X_val = validation[FEATURES]
    y_val = validation["isFraud"].astype(int).to_numpy()
    amounts = validation["amount"].to_numpy(dtype=float)

    all_results = []

    for model_name, filename in MODELS.items():
        model_path = MODEL_DIR / filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model missing: {model_path}"
            )

        model = joblib.load(model_path)
        probabilities = model.predict_proba(X_val)[:, 1]

        threshold_results = [
            evaluate_threshold(
                y_val, probabilities, amounts, threshold
            )
            for threshold in THRESHOLDS
        ]

        best = min(
            threshold_results,
            key=lambda result: result[
                "estimated_total_cost_units"
            ],
        )

        baseline = evaluate_threshold(
            y_val,
            probabilities,
            amounts,
            BASELINE_THRESHOLD,
        )

        all_results.append({
            "model": model_name,
            "average_precision_pr_auc": average_precision_score(
                y_val, probabilities
            ),
            "baseline_threshold": BASELINE_THRESHOLD,
            "baseline_cost_units": baseline[
                "estimated_total_cost_units"
            ],
            "baseline_false_positives": baseline[
                "false_positives"
            ],
            "baseline_false_negatives": baseline[
                "false_negatives"
            ],
            "best_threshold": best["threshold"],
            "best_cost_units": best[
                "estimated_total_cost_units"
            ],
            "false_positives_at_best": best[
                "false_positives"
            ],
            "false_negatives_at_best": best[
                "false_negatives"
            ],
            "precision_at_best": best["precision"],
            "recall_at_best": best["recall"],
            "f1_at_best": best["f1"],
            "estimated_savings_units": (
                baseline["estimated_total_cost_units"]
                - best["estimated_total_cost_units"]
            ),
        })

    results = pd.DataFrame(all_results)
    results.to_csv(
        REPORT_DIR / "threshold_benchmark.csv",
        index=False,
    )

    print("\n=== THRESHOLD BENCHMARK: VALIDATION ===")
    print(
        results.sort_values("best_cost_units")[
            [
                "model",
                "average_precision_pr_auc",
                "baseline_cost_units",
                "best_threshold",
                "best_cost_units",
                "false_positives_at_best",
                "false_negatives_at_best",
                "precision_at_best",
                "recall_at_best",
                "estimated_savings_units",
            ]
        ].round(4).to_string(index=False)
    )

    print(
        "\nSaved: reports/threshold_benchmark.csv"
    )
    print(
        "NOTE: Costs are assumed units, not verified USD."
    )


if __name__ == "__main__":
    main()
