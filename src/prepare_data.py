
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.data import load_config


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "processed"
REPORT_PATH = ROOT / "reports" / "split_report.json"

SOURCE_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "isFraud",
]

OUTPUT_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "log_amount",
    "amount_to_origin_balance",
    "amount_to_destination_balance",
    "origin_balance_covers_amount",
    "isFraud",
]


def engineer_features(chunk):
    """Build features using pre-transaction information only."""
    df = chunk[SOURCE_COLUMNS].copy()

    df["log_amount"] = np.log1p(df["amount"])

    df["amount_to_origin_balance"] = (
        df["amount"] / (df["oldbalanceOrg"] + 1)
    )

    df["amount_to_destination_balance"] = (
        df["amount"] / (df["oldbalanceDest"] + 1)
    )

    df["origin_balance_covers_amount"] = (
        df["oldbalanceOrg"] >= df["amount"]
    ).astype("int8")

    return df[OUTPUT_COLUMNS]


def prepare_data():
    config = load_config()

    source_path = Path(config["data"]["source_path"])
    chunk_size = config["data"]["chunk_size"]

    train_max = config["split"]["train_max_step"]
    validation_max = config["split"]["validation_max_step"]

    if train_max >= validation_max:
        raise ValueError(
            "train_max_step must be smaller than validation_max_step"
        )

    if not source_path.exists():
        raise FileNotFoundError(f"Dataset not found: {source_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "train": OUTPUT_DIR / "train.csv",
        "validation": OUTPUT_DIR / "validation.csv",
        "test": OUTPUT_DIR / "test.csv",
    }

    # Remove old generated outputs so reruns don't append duplicate data.
    for path in output_paths.values():
        path.unlink(missing_ok=True)

    stats = {
        name: {"rows": 0, "fraud": 0, "min_step": None, "max_step": None}
        for name in output_paths
    }

    for chunk in pd.read_csv(
        source_path,
        usecols=SOURCE_COLUMNS,
        chunksize=chunk_size,
    ):
        features = engineer_features(chunk)

        masks = {
            "train": features["step"] <= train_max,
            "validation": (
                (features["step"] > train_max)
                & (features["step"] <= validation_max)
            ),
            "test": features["step"] > validation_max,
        }

        for split_name, mask in masks.items():
            part = features.loc[mask]

            if part.empty:
                continue

            output_path = output_paths[split_name]

            part.to_csv(
                output_path,
                mode="a",
                header=not output_path.exists(),
                index=False,
            )

            current = stats[split_name]
            current["rows"] += len(part)
            current["fraud"] += int(part["isFraud"].sum())

            current_min = int(part["step"].min())
            current_max = int(part["step"].max())

            if current["min_step"] is None:
                current["min_step"] = current_min
            else:
                current["min_step"] = min(
                    current["min_step"], current_min
                )

            if current["max_step"] is None:
                current["max_step"] = current_max
            else:
                current["max_step"] = max(
                    current["max_step"], current_max
                )

    for split_name, values in stats.items():
        values["fraud_rate_percent"] = (
            round(values["fraud"] / values["rows"] * 100, 6)
            if values["rows"] else 0
        )

    report = {
        "source": str(source_path),
        "split_boundaries": {
            "train_max_step": train_max,
            "validation_max_step": validation_max,
        },
        "splits": stats,
        "total_rows": sum(v["rows"] for v in stats.values()),
        "total_fraud": sum(v["fraud"] for v in stats.values()),
        "features": [
            col for col in OUTPUT_COLUMNS if col != "isFraud"
        ],
        "excluded_columns": [
            "nameOrig",
            "nameDest",
            "newbalanceOrg",
            "newbalanceDest",
            "isFlaggedFraud",
        ],
    }

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n=== FEATURE PREPARATION REPORT ===")

    for split_name, values in stats.items():
        print(f"\n{split_name.upper()}")
        print(f"Rows: {values['rows']:,}")
        print(f"Fraud: {values['fraud']:,}")
        print(f"Fraud rate: {values['fraud_rate_percent']}%")
        print(
            f"Step range: {values['min_step']} to {values['max_step']}"
        )
        print(f"Output: {output_paths[split_name]}")

    print(f"\nTotal rows: {report['total_rows']:,}")
    print(f"Total fraud: {report['total_fraud']:,}")
    print(f"Report saved: {REPORT_PATH}")

    if report["total_rows"] != 6_362_620:
        raise ValueError(
            "Prepared row count differs from the audited dataset. "
            "Inspect split boundaries and source data."
        )

    if report["total_fraud"] != 8_213:
        raise ValueError(
            "Prepared fraud count differs from the audited dataset."
        )

    return report


if __name__ == "__main__":
    prepare_data()
