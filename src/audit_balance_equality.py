
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"


def main():
    all_reports = []

    for split in ["train", "validation", "test"]:
        path = DATA_DIR / f"{split}.csv"

        df = pd.read_csv(
            path,
            usecols=[
                "type",
                "amount",
                "oldbalanceOrg",
                "isFraud",
            ],
        )

        # Is the transaction amount exactly equal to the
        # origin account's balance before the transaction?
        df["exact_balance_match"] = (
            df["amount"] == df["oldbalanceOrg"]
        ).astype("int8")

        # Also check for a tiny floating-point difference.
        df["near_balance_match"] = (
            (df["amount"] - df["oldbalanceOrg"]).abs() <= 0.01
        ).astype("int8")

        report = (
            df.groupby(["type", "isFraud"], observed=True)
            .agg(
                transactions=("isFraud", "size"),
                exact_match_rate=("exact_balance_match", "mean"),
                near_match_rate=("near_balance_match", "mean"),
            )
            .reset_index()
        )

        report.insert(0, "split", split)
        report["isFraud"] = report["isFraud"].map(
            {0: "Legitimate", 1: "Fraud"}
        )

        all_reports.append(report)

        print(f"\n{'=' * 65}")
        print(f"{split.upper()} — BALANCE EQUALITY AUDIT")

        print(
            report.to_string(
                index=False,
                formatters={
                    "exact_match_rate": lambda x: f"{x:.4%}",
                    "near_match_rate": lambda x: f"{x:.4%}",
                },
            )
        )

        del df

    output_path = ROOT / "reports" / "balance_equality_audit.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.concat(all_reports, ignore_index=True).to_csv(
        output_path, index=False
    )

    print(f"\nAudit report saved to: {output_path}")


if __name__ == "__main__":
    main()
