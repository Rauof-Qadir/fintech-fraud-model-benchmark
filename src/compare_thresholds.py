
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


def evaluate_thresholds(y_true, probabilities, amounts, thresholds):
    """Evaluate many thresholds efficiently using sorted probabilities."""
    order = np.argsort(probabilities)
    sorted_probabilities = probabilities[order]
    sorted_y = y_true[order]
    sorted_amounts = amounts[order]

    n = len(y_true)
    total_fraud = int(y_true.sum())
    prefix_fraud = np.concatenate(([0], np.cumsum(sorted_y, dtype=np.int64)))
    prefix_missed_amount = np.concatenate((
        [0.0],
        np.cumsum(sorted_amounts * sorted_y, dtype=np.float64),
    ))

    results = []

    for threshold in thresholds:
        idx = int(np.searchsorted(
            sorted_probabilities, threshold, side="left"
        ))

        tp = int(total_fraud - prefix_fraud[idx])
        fp = int((n - idx) - tp)
        fn = int(total_fraud - tp)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / total_fraud if total_fraud else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) else 0.0
        )

        fp_cost = fp * FP_COST
        missed_amount = float(prefix_missed_amount[idx])

        results.append({
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "false_positive_cost_units": fp_cost,
            "missed_fraud_amount_units": missed_amount,
            "estimated_total_cost_units": fp_cost + missed_amount,
        })

    return results


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

        threshold_results = evaluate_thresholds(
            y_val, probabilities, amounts, THRESHOLDS
        )

        best = min(
            threshold_results,
            key=lambda result: result[
                "estimated_total_cost_units"
            ],
        )

        baseline = evaluate_thresholds(
            y_val,
            probabilities,
            amounts,
            [BASELINE_THRESHOLD],
        )[0]

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
