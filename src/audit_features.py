
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

SPLITS = ["train", "validation", "test"]

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

TARGET = "isFraud"

# These columns are unavailable before a transaction completes,
# or are not approved for this model's current feature set.
FORBIDDEN_FEATURES = {
    "newbalanceOrig",
    "newbalanceDest",
    "isFlaggedFraud",
    "nameOrig",
    "nameDest",
}


def main():
    reports = []

    for split in SPLITS:
        path = DATA_DIR / f"{split}.csv"
        df = pd.read_csv(path)

        missing = set(FEATURES + [TARGET]) - set(df.columns)
        leaked = set(FEATURES) & FORBIDDEN_FEATURES

        if missing:
            raise ValueError(
                f"{split}: missing columns: {sorted(missing)}"
            )

        if leaked:
            raise ValueError(
                f"{split}: forbidden features found: {sorted(leaked)}"
            )

        fraud_rate = df[TARGET].mean()

        print(f"\n{'=' * 55}")
        print(f"{split.upper()} DATA")
        print(f"Rows: {len(df):,}")
        print(f"Fraud cases: {int(df[TARGET].sum()):,}")
        print(f"Fraud rate: {fraud_rate:.4%}")
        print(f"Step range: {df['step'].min()} - {df['step'].max()}")
        print("Forbidden features in model input: NONE")

        print("\nFraud rate by transaction type:")
        type_report = (
            df.groupby("type", observed=True)[TARGET]
            .agg(["count", "sum", "mean"])
            .rename(columns={
                "count": "transactions",
                "sum": "fraud_cases",
                "mean": "fraud_rate",
            })
            .sort_values("transactions", ascending=False)
        )

        print(type_report.to_string(
            formatters={"fraud_rate": lambda x: f"{x:.4%}"}
        ))

        reports.append({
            "split": split,
            "rows": len(df),
            "fraud_cases": int(df[TARGET].sum()),
            "fraud_rate": fraud_rate,
            "min_step": int(df["step"].min()),
            "max_step": int(df["step"].max()),
        })

    summary = pd.DataFrame(reports)
    output = ROOT / "reports" / "feature_distribution_audit.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)

    print("\n=== SPLIT DISTRIBUTION SUMMARY ===")
    print(summary.to_string(index=False))
    print(f"\nSaved: {output}")
    print(
        "\nNote: passing these checks does not prove that "
        "all forms of leakage are absent."
    )


if __name__ == "__main__":
    main()
