
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
THRESHOLD_REPORT_PATH = REPORT_DIR / "threshold_benchmark.csv"

FP_COSTS = [5, 25, 50, 100]
MISSED_LOSS_MULTIPLIERS = [0.10, 0.25, 0.50, 1.00]

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
        if not pd.notna(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid threshold for {name}: {value}")
        thresholds[name] = value

    return thresholds


def main():
    test = pd.read_csv(DATA_DIR / "test.csv")

    y_true = test["isFraud"].astype(int).to_numpy()
    amounts = test["amount"].to_numpy(dtype=float)

    predictions = {
        "Exact-match rule": (
            test["amount"] == test["oldbalanceOrg"]
        ).to_numpy(),
        "Balance-coverage rule": (
            test["type"].isin(["CASH_OUT", "TRANSFER"])
            & (test["oldbalanceOrg"] >= test["amount"])
        ).to_numpy(),
    }

    X_test = test[FEATURES]

    # Read thresholds selected using validation data only.
    thresholds = load_validation_thresholds()
    for name, config in MODELS.items():
        model = joblib.load(MODEL_DIR / config["file"])
        probabilities = model.predict_proba(X_test)[:, 1]
        threshold = thresholds[name]

        predictions[name] = probabilities >= threshold

        print(
            f"Loaded {name}: "
            f"validation threshold={threshold:.2f}"
        )

    base_metrics = []

    for strategy, predicted in predictions.items():
        tn, fp, fn, tp = confusion_matrix(
            y_true, predicted.astype(int), labels=[0, 1]
        ).ravel()

        missed = (y_true == 1) & (~predicted)

        base_metrics.append({
            "strategy": strategy,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
            "missed_fraud_amount_units": float(
                amounts[missed].sum()
            ),
        })

    base_df = pd.DataFrame(base_metrics)
    all_rows = []

    for _, metric in base_df.iterrows():
        for fp_cost in FP_COSTS:
            for loss_multiplier in MISSED_LOSS_MULTIPLIERS:
                investigation_cost = (
                    metric["false_positives"] * fp_cost
                )
                estimated_missed_loss = (
                    metric["missed_fraud_amount_units"]
                    * loss_multiplier
                )

                all_rows.append({
                    **metric.to_dict(),
                    "fp_cost_per_case": fp_cost,
                    "missed_loss_multiplier": loss_multiplier,
                    "investigation_cost_units": investigation_cost,
                    "estimated_missed_loss_units": estimated_missed_loss,
                    "estimated_total_cost_units": (
                        investigation_cost + estimated_missed_loss
                    ),
                })

    results = pd.DataFrame(all_rows)

    # Identify the lowest-cost strategy for every scenario.
    results["scenario_min_cost"] = results.groupby(
        ["fp_cost_per_case", "missed_loss_multiplier"]
    )["estimated_total_cost_units"].transform("min")

    results["is_lowest_cost"] = (
        results["estimated_total_cost_units"]
        <= results["scenario_min_cost"] + 1e-9
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    detail_path = REPORT_DIR / "all_strategy_cost_sensitivity.csv"
    summary_path = REPORT_DIR / "best_strategy_by_cost_scenario.csv"

    results.to_csv(detail_path, index=False)

    best = results[results["is_lowest_cost"]].copy()
    best = best.sort_values(
        ["fp_cost_per_case", "missed_loss_multiplier", "strategy"]
    )
    best.to_csv(summary_path, index=False)

    print("\nTEST SET")
    print("=" * 72)
    print(f"Transactions: {len(test):,}")
    print(f"Fraud cases: {int(y_true.sum()):,}")
    print("\nBase detection metrics:")
    print(
        base_df.to_string(
            index=False,
            formatters={
                "missed_fraud_amount_units": lambda x: f"{x:,.2f}"
            },
        )
    )

    print("\nLOWEST-COST STRATEGY BY SCENARIO")
    print("=" * 72)
    print(
        best[
            [
                "fp_cost_per_case",
                "missed_loss_multiplier",
                "strategy",
                "false_positives",
                "false_negatives",
                "estimated_total_cost_units",
            ]
        ].to_string(
            index=False,
            formatters={
                "missed_loss_multiplier": lambda x: f"{x:.0%}",
                "estimated_total_cost_units": lambda x: f"{x:,.2f}",
            },
        )
    )

    print(f"\nDetailed report: {detail_path}")
    print(f"Best-strategy report: {summary_path}")
    print(
        "\nNOTE: Costs are illustrative dataset units, not verified currency "
        "or actual bank losses."
    )
    print(
        "Thresholds were selected on validation data; this analysis does "
        "not retune them."
    )


if __name__ == "__main__":
    main()
