
from collections import Counter
from pathlib import Path
import json

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
REPORT_PATH = ROOT / "reports" / "data_quality_report.json"


def load_config():
    """Load project settings from YAML."""
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def audit_dataset():
    config = load_config()
    data_path = Path(config["data"]["source_path"])
    chunk_size = config["data"]["chunk_size"]
    target = config["data"]["target_column"]
    required = config["validation"]["required_columns"]
    allowed_types = set(
        config["validation"]["allowed_transaction_types"]
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Check source_path in configs/config.yaml"
        )

    # Validate the CSV header before processing millions of rows.
    header = pd.read_csv(data_path, nrows=0).columns.tolist()

    missing_columns = sorted(set(required) - set(header))
    unexpected_columns = sorted(set(header) - set(required))

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    report = {
        "dataset": str(data_path),
        "status": "running",
        "rows": 0,
        "columns": len(header),
        "missing_values": Counter(),
        "transaction_types": Counter(),
        "fraud_labels": Counter(),
        "invalid_target_rows": 0,
        "invalid_transaction_type_rows": 0,
        "negative_numeric_values": Counter(),
        "zero_amount_rows": 0,
        "duplicate_rows_within_chunks": 0,
        "missing_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
        "amount_min": None,
        "amount_max": None,
    }

    numeric_columns = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud",
    ]

    min_values = {col: float("inf") for col in numeric_columns}
    max_values = {col: float("-inf") for col in numeric_columns}

    for chunk in pd.read_csv(data_path, chunksize=chunk_size):
        report["rows"] += len(chunk)

        for col, count in chunk.isna().sum().items():
            report["missing_values"][col] += int(count)

        for value, count in chunk["type"].value_counts().items():
            report["transaction_types"][str(value)] += int(count)

        for value, count in chunk[target].value_counts(
            dropna=False
        ).items():
            report["fraud_labels"][str(value)] += int(count)

        report["invalid_target_rows"] += int(
            (~chunk[target].isin([0, 1])).sum()
        )

        report["invalid_transaction_type_rows"] += int(
            (~chunk["type"].isin(allowed_types)).sum()
        )

        report["zero_amount_rows"] += int(
            (chunk["amount"] == 0).sum()
        )

        report["duplicate_rows_within_chunks"] += int(
            chunk.duplicated().sum()
        )

        for col in numeric_columns:
            values = pd.to_numeric(chunk[col], errors="coerce")

            report["negative_numeric_values"][col] += int(
                (values < 0).sum()
            )

            if values.notna().any():
                min_values[col] = min(
                    min_values[col], float(values.min())
                )
                max_values[col] = max(
                    max_values[col], float(values.max())
                )

    report["missing_values"] = dict(report["missing_values"])
    report["transaction_types"] = dict(report["transaction_types"])
    report["fraud_labels"] = dict(report["fraud_labels"])
    report["negative_numeric_values"] = dict(
        report["negative_numeric_values"]
    )

    report["numeric_ranges"] = {
        col: {
            "min": min_values[col] if min_values[col] != float("inf") else None,
            "max": max_values[col] if max_values[col] != float("-inf") else None,
        }
        for col in numeric_columns
    }

    report["amount_min"] = report["numeric_ranges"]["amount"]["min"]
    report["amount_max"] = report["numeric_ranges"]["amount"]["max"]

    fraud_count = report["fraud_labels"].get("1", 0)
    total_rows = report["rows"]

    report["fraud_rate_percent"] = (
        round(fraud_count / total_rows * 100, 6)
        if total_rows else 0
    )

    report["status"] = (
        "completed"
        if report["invalid_target_rows"] == 0
        and report["invalid_transaction_type_rows"] == 0
        else "review_required"
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, default=str)

    print("\n=== FINTECH DATA QUALITY REPORT ===")
    print(f"Rows processed: {report['rows']:,}")
    print(f"Columns: {report['columns']}")
    print(f"Fraud labels: {report['fraud_labels']}")
    print(f"Fraud rate: {report['fraud_rate_percent']}%")
    print(f"Transaction types: {report['transaction_types']}")
    print(f"Missing values: {report['missing_values']}")
    print(f"Zero-amount transactions: {report['zero_amount_rows']:,}")
    print(
        "Negative numeric values:",
        report["negative_numeric_values"],
    )
    print(
        "Duplicate rows within chunks:",
        report["duplicate_rows_within_chunks"],
    )
    print(f"Status: {report['status']}")
    print(f"Report saved to: {REPORT_PATH}")

    return report


if __name__ == "__main__":
    audit_dataset()
