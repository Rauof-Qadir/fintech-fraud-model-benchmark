
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

COLUMNS = [
    "type",
    "amount",
    "oldbalanceOrg",
    "amount_to_origin_balance",
    "origin_balance_covers_amount",
    "isFraud",
]


def main():
    for split in ["train", "validation", "test"]:
        path = DATA_DIR / f"{split}.csv"

        print(f"\n{'=' * 65}")
        print(f"{split.upper()} — FEATURE PATTERN AUDIT")

        # Read only the columns needed for this audit.
        df = pd.read_csv(path, usecols=COLUMNS)

        print("\n1. Origin balance covers transaction amount")
        covers_report = (
            df.groupby(["type", "isFraud"], observed=True)
            ["origin_balance_covers_amount"]
            .agg(["count", "mean"])
            .reset_index()
        )
        covers_report["isFraud"] = covers_report["isFraud"].map(
            {0: "Legitimate", 1: "Fraud"}
        )
        covers_report = covers_report.rename(columns={
            "count": "transactions",
            "mean": "fraction_balance_covers_amount",
        })

        print(covers_report.to_string(
            index=False,
            formatters={
                "fraction_balance_covers_amount":
                    lambda value: f"{value:.4%}"
            },
        ))

        print("\n2. Origin amount ratio by transaction type and label")
        ratio_report = (
            df.groupby(["type", "isFraud"], observed=True)
            ["amount_to_origin_balance"]
            .agg(["count", "median", "mean"])
            .reset_index()
        )
        ratio_report["isFraud"] = ratio_report["isFraud"].map(
            {0: "Legitimate", 1: "Fraud"}
        )
        ratio_report = ratio_report.rename(columns={
            "count": "transactions",
            "median": "median_ratio",
            "mean": "mean_ratio",
        })

        print(ratio_report.to_string(
            index=False,
            formatters={
                "median_ratio": lambda value: f"{value:.4f}",
                "mean_ratio": lambda value: f"{value:.4f}",
            },
        ))

        del df

    print(
        "\nInterpretation: differences show association, not causation "
        "and not proof of leakage."
    )


if __name__ == "__main__":
    main()
