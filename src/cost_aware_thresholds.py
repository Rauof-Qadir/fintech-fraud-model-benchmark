from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

FP_COSTS = [5, 25, 50, 100]
MISSED_LOSS_MULTIPLIERS = [0.10, 0.25, 0.50, 1.00]
THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)

FEATURES = [
    "step", "type", "amount", "oldbalanceOrg", "oldbalanceDest",
    "log_amount", "amount_to_origin_balance",
    "amount_to_destination_balance", "origin_balance_covers_amount",
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
}


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    validation = pd.read_csv(DATA / "validation.csv")

    X = validation[FEATURES]
    y = validation["isFraud"].astype(int).to_numpy()
    amounts = validation["amount"].to_numpy(dtype=float)

    total_fraud = int(y.sum())
    all_rows = []

    for model_name, filename in MODEL_FILES.items():
        model_path = MODELS / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model missing: {model_path}")

        model = joblib.load(model_path)
        probabilities = model.predict_proba(X)[:, 1]

        # Sort once; use cumulative sums to evaluate thresholds efficiently.
        order = np.argsort(probabilities)
        p = probabilities[order]
        sy = y[order]
        sa = amounts[order]

        prefix_fraud = np.concatenate(([0], np.cumsum(sy)))
        prefix_fraud_amount = np.concatenate(
            ([0.0], np.cumsum(sa * sy, dtype=np.float64))
        )

        for threshold in THRESHOLDS:
            idx = np.searchsorted(p, threshold, side="left")
            tp = total_fraud - int(prefix_fraud[idx])
            fp = (len(y) - idx) - tp
            fn = total_fraud - tp

            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / total_fraud if total_fraud else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )

            missed_amount = float(prefix_fraud_amount[idx])

            for fp_cost in FP_COSTS:
                for multiplier in MISSED_LOSS_MULTIPLIERS:
                    cost = fp * fp_cost + missed_amount * multiplier

                    all_rows.append({
                        "model": model_name,
                        "threshold": float(threshold),
                        "fp_cost_per_case": fp_cost,
                        "missed_loss_multiplier": multiplier,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "true_positives": tp,
                        "false_positives": fp,
                        "false_negatives": fn,
                        "missed_fraud_amount_units": missed_amount,
                        "estimated_total_cost_units": cost,
                    })

    results = pd.DataFrame(all_rows)

    # One best validation threshold for each model and cost scenario.
    best = (
        results.sort_values("estimated_total_cost_units")
        .groupby(
            ["model", "fp_cost_per_case", "missed_loss_multiplier"],
            as_index=False,
        )
        .first()
    )

    all_path = REPORTS / "validation_cost_threshold_grid.csv"
    best_path = REPORTS / "validation_best_cost_thresholds.csv"

    results.to_csv(all_path, index=False)
    best.to_csv(best_path, index=False)

    print("\n=== BEST THRESHOLD PER VALIDATION COST SCENARIO ===")
    print(
        best[
            [
                "model", "fp_cost_per_case", "missed_loss_multiplier",
                "threshold", "precision", "recall", "false_positives",
                "false_negatives", "estimated_total_cost_units",
            ]
        ].to_string(
            index=False,
            formatters={
                "threshold": lambda x: f"{x:.2f}",
                "precision": lambda x: f"{x:.4f}",
                "recall": lambda x: f"{x:.4f}",
                "estimated_total_cost_units": lambda x: f"{x:,.2f}",
            },
        )
    )

    print(f"\nAll validation combinations: {all_path}")
    print(f"Best validation thresholds: {best_path}")
    print(
        "\nThresholds were selected on validation data only. "
        "Costs are illustrative units, not verified currency."
    )


if __name__ == "__main__":
    main()
