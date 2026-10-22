
import pandas as pd


def validate_schema(columns, required_columns):
    """Return required columns that are missing."""
    return sorted(set(required_columns) - set(columns))


def inspect_chunk(chunk, target_column, allowed_types):
    """Inspect a DataFrame without modifying its rows."""
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
        "duplicate_rows_within_chunk": int(
            chunk.duplicated().sum()
        ),
    }

    return findings


def get_quality_status(findings):
    """Separate critical validation failures from warnings."""
    critical_checks = [
        "missing_values",
        "invalid_target_rows",
        "invalid_transaction_type_rows",
        "negative_amount_rows",
    ]

    if any(findings.get(key, 0) > 0 for key in critical_checks):
        return "review_required"

    return "pass"
