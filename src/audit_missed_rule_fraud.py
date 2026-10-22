
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "data" / "processed" / "test.csv"
REPORT_DIR = ROOT / "reports"


def main():
    df = pd.read_csv(
        TEST_FILE,
        usecols=[
            "step",
            "type",
            "amount",
            "oldbalanceOrg",
            "oldbalanceDest",
            "isFraud",
        ],
    )

    # Fraud transactions not detected by the exact-match rule.
    missed = df[
        (df["isFraud"] == 1)
        & (df["amount"] != df["oldbalanceOrg"])
    ].copy()

    missed = missed.sort_values("amount", ascending=False)

    print("\nMISSED FRAUD — EXACT-MATCH RULE")
    print("-" * 65)
    print(f"Missed fraud cases: {len(missed):,}")
    print(f"Combined transaction amount: {missed['amount'].sum():,.2f}")
    print("\nTop 20 missed fraud transactions:")
    print(
        missed.head(20).to_string(
            index=False,
            formatters={
                "amount": lambda x: f"{x:,.2f}",
                "oldbalanceOrg": lambda x: f"{x:,.2f}",
                "oldbalanceDest": lambda x: f"{x:,.2f}",
            },
        )
    )

    output = REPORT_DIR / "missed_rule_fraud.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    missed.to_csv(output, index=False)

    print(f"\nFull report saved to: {output}")


if __name__ == "__main__":
    main()
