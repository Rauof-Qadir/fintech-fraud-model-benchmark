from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

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
    validation_choices_path = REPORTS / "validation_best_cost_thresholds.csv"
    if not validation_choices_path.exists():
        raise FileNotFoundError(
            f"Run src/cost_aware_thresholds.py first: {validation_choices_path}"
        )

    choices = pd.read_csv(validation_choices_path)
    test = pd.read_csv(DATA / "test.csv")

    X = test[FEATURES]
    y = test["isFraud"].astype(int).to_numpy()
    amounts = test["amount"].to_numpy(dtype=float)

    # Cache each model's test probabilities; do not tune on test outcomes.
    model_results = {}
    for name, filename in MODEL_FILES.items():
        model = joblib.load(MODELS / filename)
        probabilities = model.predict_proba(X)[:, 1]
        model_results[name] = {
            "probabilities": probabilities,
            "pr_auc": average_precision_score(y, probabilities),
        }

    rows = []

    for _, choice in choices.iterrows():
        name = choice["model"]
        threshold = float(choice["threshold"])
        fp_cost = float(choice["fp_cost_per_case"])
        multiplier = float(choice["missed_loss_multiplier"])

        probabilities = model_results[name]["probabilities"]
        predicted = probabilities >= threshold

        tp = int(np.sum(predicted & (y == 1)))
        fp = int(np.sum(predicted & (y == 0)))
        fn = int(np.sum((~predicted) & (y == 1)))
        tn = int(np.sum((~predicted) & (y == 0)))

        missed_amount = float(amounts[(y == 1) & (~predicted)].sum())
        estimated_cost = fp * fp_cost + missed_amount * multiplier

        rows.append({
            "model": name,
            "fp_cost_per_case": fp_cost,
            "missed_loss_multiplier": multiplier,
            "validation_selected_threshold": threshold,
            "test_pr_auc": model_results[name]["pr_auc"],
            "test_true_positives": tp,
            "test_false_positives": fp,
            "test_false_negatives": fn,
            "test_true_negatives": tn,
            "test_missed_fraud_amount_units": missed_amount,
            "test_estimated_cost_units": estimated_cost,
        })

    results = pd.DataFrame(rows)

    # Compare models only within the same cost scenario.
    results["scenario_min_cost"] = results.groupby(
        ["fp_cost_per_case", "missed_loss_multiplier"]
    )["test_estimated_cost_units"].transform("min")

    results["lowest_test_cost_model"] = (
        results["test_estimated_cost_units"]
        <= results["scenario_min_cost"] + 1e-9
    )

    output = REPORTS / "test_cost_aware_evaluation.csv"
    results.to_csv(output, index=False)

    winners = results[results["lowest_test_cost_model"]].copy()
    winners = winners.sort_values(
        ["fp_cost_per_case", "missed_loss_multiplier", "model"]
    )
    winner_path = REPORTS / "test_cost_scenario_winners.csv"
    winners.to_csv(winner_path, index=False)

    print(f"Test rows: {len(test):,}")
    print(f"Actual fraud cases: {int(y.sum()):,}")
    print("\n=== TEST RESULTS USING VALIDATION-SELECTED THRESHOLDS ===")
    print(
        results[
            [
                "model", "fp_cost_per_case", "missed_loss_multiplier",
                "validation_selected_threshold", "test_false_positives",
                "test_false_negatives", "test_estimated_cost_units",
                "lowest_test_cost_model",
            ]
        ].to_string(
            index=False,
            formatters={
                "validation_selected_threshold": lambda x: f"{x:.2f}",
                "test_estimated_cost_units": lambda x: f"{x:,.2f}",
            },
        )
    )

    print(f"\nFull report: {output}")
    print(f"Scenario winners: {winner_path}")
    print(
        "\nCosts are illustrative units, not verified currency. "
        "Test results are for evaluation, not threshold selection."
    )


if __name__ == "__main__":
    main()
