
import pandas as pd


def validate_schema(columns, required_columns):
    """Return required columns that are missing."""
    return sorted(set(required_columns) - set(columns))


def inspect_chunk(chunk, target_column, allowed_types):
    """Inspect data without deleting or modifying any transactions."""
    numeric_columns = chunk.select_dtypes(include="number").columns

    negative_numeric_rows = int(
        chunk[list(numeric_columns)].lt(0).any(axis=1).sum()
    )

    findings = {
        "missing_values": int(chunk.isna().sum().sum()),
        "invalid_target_rows": int(
            (~chunk[target_column].isin([0, 1])).sum()
        ),
        "invalid_transaction_type_rows": int(
            (~chunk["type"].isin(allowed_types)).sum()
        ),
        "zero_amount_rows": int((chunk["amount"] == 0).sum()),
        "negative_amount_rows": int((chunk["amount"] < 0).sum()),
        "negative_numeric_rows": negative_numeric_rows,
        "duplicate_rows_within_chunk": int(chunk.duplicated().sum()),
    }

    return findings


def get_quality_status(findings):
    """Return pass only when critical validation checks are clean."""
    critical_checks = [
        "missing_values",
        "invalid_target_rows",
        "invalid_transaction_type_rows",
        "negative_numeric_rows",
    ]

    if any(findings.get(key, 0) > 0 for key in critical_checks):
        return "review_required"

    return "pass"
