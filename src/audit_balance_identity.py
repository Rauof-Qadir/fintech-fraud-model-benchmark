from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
SPLITS = ["train", "validation", "test"]

for split in SPLITS:
    path = DATA_DIR / f"{split}.csv"
    if not path.exists():
        print(f"\nMissing file: {path}")
        continue

    counts = {
        "rows": 0,
        "fraud": 0,
        "fraud_full_drain": 0,
        "legit": 0,
        "legit_full_drain": 0,
        "fraud_transfer_cashout": 0,
        "fraud_transfer_cashout_full_drain": 0,
    }

    for chunk in pd.read_csv(
        path,
        usecols=["type", "amount", "oldbalanceOrg", "isFraud"],
        chunksize=200_000,
    ):
        fraud = chunk["isFraud"].eq(1)
        legit = ~fraud
        full_drain = np.isclose(
            chunk["amount"].to_numpy(),
            chunk["oldbalanceOrg"].to_numpy(),
            rtol=0,
            atol=0.01,
        )
        relevant_type = chunk["type"].isin(["TRANSFER", "CASH_OUT"])

        counts["rows"] += len(chunk)
        counts["fraud"] += int(fraud.sum())
        counts["fraud_full_drain"] += int((fraud & full_drain).sum())
        counts["legit"] += int(legit.sum())
        counts["legit_full_drain"] += int((legit & full_drain).sum())
        counts["fraud_transfer_cashout"] += int(
            (fraud & relevant_type).sum()
        )
        counts["fraud_transfer_cashout_full_drain"] += int(
            (fraud & relevant_type & full_drain).sum()
        )

    fraud_rate = (
        counts["fraud_full_drain"] / counts["fraud"]
        if counts["fraud"] else 0
    )
    legit_rate = (
        counts["legit_full_drain"] / counts["legit"]
        if counts["legit"] else 0
    )
    relevant_rate = (
        counts["fraud_transfer_cashout_full_drain"]
        / counts["fraud_transfer_cashout"]
        if counts["fraud_transfer_cashout"] else 0
    )

    print(f"\n===== {split.upper()} =====")
    print(f"Rows: {counts['rows']:,}")
    print(f"Fraud rows: {counts['fraud']:,}")
    print(
        f"Fraud where amount ~= oldbalanceOrg: "
        f"{counts['fraud_full_drain']:,} ({fraud_rate:.2%})"
    )
    print(
        f"Legitimate rows where amount ~= oldbalanceOrg: "
        f"{counts['legit_full_drain']:,} ({legit_rate:.4%})"
    )
    print(
        f"Fraud in TRANSFER/CASH_OUT: "
        f"{counts['fraud_transfer_cashout']:,}"
    )
    print(
        f"Those fraud rows matching balance: "
        f"{counts['fraud_transfer_cashout_full_drain']:,} "
        f"({relevant_rate:.2%})"
    )
