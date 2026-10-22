from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
THRESHOLD_FILE = REPORTS / "threshold_benchmark.csv"

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

FP_COST = 5.0


def load_thresholds():
    table = pd.read_csv(THRESHOLD_FILE)
    required = {"model", "best_threshold"}

    if not required.issubset(table.columns):
        raise ValueError(f"Threshold report needs columns: {required}")

    if table["model"].duplicated().any():
        raise ValueError("Duplicate model names in threshold report.")

    thresholds = table.set_index("model")["best_threshold"].to_dict()

    for name in MODEL_FILES:
        if name not in thresholds:
            raise ValueError(f"Missing validation threshold for {name}")

        value = float(thresholds[name])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid threshold for {name}: {value}")

        thresholds[name] = value

    return thresholds


def evaluate(name, predicted, y_true, amounts, probabilities=None):
    predicted = np.asarray(predicted, dtype=bool)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predicted.astype(int), labels=[0, 1]
    ).ravel()

    missed_amount = float(amounts[(y_true == 1) & (~predicted)].sum())

    row = {
        "strategy": name,
        "precision": precision_score(
            y_true, predicted, zero_division=0
        ),
        "recall": recall_score(
            y_true, predicted, zero_division=0
        ),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "missed_fraud_amount_units": missed_amount,
        "estimated_total_cost_units": fp * FP_COST + missed_amount,
    }

    if probabilities is not None:
        row["pr_auc"] = average_precision_score(y_true, probabilities)
    else:
        row["pr_auc"] = np.nan

    return row


def main():
    test = pd.read_csv(DATA / "test.csv")
    y = test["isFraud"].astype(int).to_numpy()
    amounts = test["amount"].to_numpy(dtype=float)

    # Fixed rules: no threshold tuning on test data.
    exact = test["amount"].eq(test["oldbalanceOrg"]).to_numpy()

    tolerant = np.isclose(
        test["amount"].to_numpy(dtype=float),
        test["oldbalanceOrg"].to_numpy(dtype=float),
        rtol=0,
        atol=0.01,
    )

    relevant_type = test["type"].isin(["TRANSFER", "CASH_OUT"]).to_numpy()

    predictions = {
        "Exact balance match": exact,
        "Balance match (tolerance 0.01)": tolerant,
        "Balance match + TRANSFER/CASH_OUT": tolerant & relevant_type,
    }

    thresholds = load_thresholds()
    X = test[FEATURES]

    for name, filename in MODEL_FILES.items():
        model = joblib.load(MODELS / filename)
        probabilities = model.predict_proba(X)[:, 1]
        predictions[name] = probabilities >= thresholds[name]

        print(
            f"{name}: validation threshold = "
            f"{thresholds[name]:.4f}"
        )

    rows = []
    for name, predicted in predictions.items():
        probabilities = None

        if name in MODEL_FILES:
            model = joblib.load(MODELS / MODEL_FILES[name])
            probabilities = model.predict_proba(X)[:, 1]

        rows.append(evaluate(name, predicted, y, amounts, probabilities))

    results = pd.DataFrame(rows).sort_values(
        "estimated_total_cost_units"
    )

    REPORTS.mkdir(parents=True, exist_ok=True)
    output = REPORTS / "balance_rule_vs_ml.csv"
    results.to_csv(output, index=False)

    print(f"\nTest rows: {len(test):,}")
    print(f"Actual fraud cases: {int(y.sum()):,}")
    print("\n=== BALANCE RULE VS ML ===")
    print(
        results[
            [
                "strategy", "precision", "recall", "f1",
                "true_positives", "false_positives",
                "false_negatives", "estimated_total_cost_units",
            ]
        ].to_string(
            index=False,
            formatters={
                "precision": lambda x: f"{x:.4f}",
                "recall": lambda x: f"{x:.4f}",
                "f1": lambda x: f"{x:.4f}",
                "estimated_total_cost_units": lambda x: f"{x:,.2f}",
            },
        )
    )

    print(f"\nSaved report: {output}")
    print(
        "\nNOTE: Costs are illustrative units, not verified currency. "
        "Rules and models are evaluated on synthetic data."
    )


if __name__ == "__main__":
    main()
