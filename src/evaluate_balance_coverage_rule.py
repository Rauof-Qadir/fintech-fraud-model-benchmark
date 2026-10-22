
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "data" / "processed" / "test.csv"
REPORT_DIR = ROOT / "reports"

FP_COST = 5


def main():
    df = pd.read_csv(
        TEST_FILE,
        usecols=[
            "type",
            "amount",
            "oldbalanceOrg",
            "isFraud",
        ],
    )

    y_true = df["isFraud"].astype("int8")

    # Broader rule: risky transaction type AND sufficient
    # pre-transaction origin balance.
    y_pred = (
        df["type"].isin(["CASH_OUT", "TRANSFER"])
        & (df["oldbalanceOrg"] >= df["amount"])
    ).astype("int8")

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    missed_amount = df.loc[
        (y_true == 1) & (y_pred == 0), "amount"
    ].sum()

    fp_cost = int(fp) * FP_COST
    total_cost = missed_amount + fp_cost

    results = {
        "transactions": len(df),
        "actual_fraud": int(y_true.sum()),
        "flagged": int(y_pred.sum()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "missed_fraud_amount": float(missed_amount),
        "false_positive_cost": float(fp_cost),
        "estimated_total_cost": float(total_cost),
    }

    print("\nBALANCE-COVERAGE RULE — TEST RESULTS")
    print("-" * 48)

    for key, value in results.items():
        if key in {"precision", "recall", "f1"}:
            print(f"{key}: {value:.4%}")
        elif isinstance(value, float):
            print(f"{key}: {value:,.2f}")
        else:
            print(f"{key}: {value:,}")

    print("\nAssumptions: FP cost = 5 units; missed fraud")
    print("cost = full transaction amount. Currency is unverified.")

    output = REPORT_DIR / "balance_coverage_rule_benchmark.csv"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([results]).to_csv(output, index=False)

    print(f"\nReport saved to: {output}")


if __name__ == "__main__":
    main()
