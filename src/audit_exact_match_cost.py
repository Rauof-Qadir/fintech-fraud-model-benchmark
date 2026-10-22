
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "data" / "processed" / "test.csv"

FALSE_POSITIVE_COST = 5


def main():
    df = pd.read_csv(
        TEST_FILE,
        usecols=[
            "amount",
            "oldbalanceOrg",
            "isFraud",
        ],
    )

    actual_fraud = df["isFraud"].eq(1)
    predicted_fraud = df["amount"].eq(df["oldbalanceOrg"])

    true_positive = predicted_fraud & actual_fraud
    false_positive = predicted_fraud & ~actual_fraud
    false_negative = ~predicted_fraud & actual_fraud
    true_negative = ~predicted_fraud & ~actual_fraud

    # Assumption: a missed fraud costs the full transaction amount.
    missed_fraud_amount = df.loc[
        false_negative, "amount"
    ].sum()

    false_positive_cost = (
        int(false_positive.sum()) * FALSE_POSITIVE_COST
    )

    total_cost = missed_fraud_amount + false_positive_cost

    print("\nEXACT-MATCH RULE — TEST BUSINESS COST")
    print("-" * 45)
    print(f"True positives: {int(true_positive.sum()):,}")
    print(f"False positives: {int(false_positive.sum()):,}")
    print(f"False negatives: {int(false_negative.sum()):,}")
    print(f"True negatives: {int(true_negative.sum()):,}")
    print(f"Missed fraud amount: {missed_fraud_amount:,.2f}")
    print(f"False-positive cost: {false_positive_cost:,.2f}")
    print(f"Total estimated cost: {total_cost:,.2f}")

    print("\nIMPORTANT:")
    print("Costs are in assumed dataset units, not verified USD.")
    print("This assumes every missed fraud loses its full amount.")

    output = ROOT / "reports" / "exact_match_rule_cost.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"Test transactions: {len(df)}\n"
        f"True positives: {int(true_positive.sum())}\n"
        f"False positives: {int(false_positive.sum())}\n"
        f"False negatives: {int(false_negative.sum())}\n"
        f"True negatives: {int(true_negative.sum())}\n"
        f"Missed fraud amount: {missed_fraud_amount:.2f}\n"
        f"False-positive cost: {false_positive_cost:.2f}\n"
        f"Total estimated cost: {total_cost:.2f}\n",
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output}")


if __name__ == "__main__":
    main()
