
import pandas as pd

from src.validation import (
    validate_schema,
    inspect_chunk,
    get_quality_status,
)


ALLOWED_TYPES = {
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
}


def make_transactions():
    return pd.DataFrame({
        "type": ["TRANSFER", "PAYMENT", "CASH_OUT"],
        "amount": [5000.0, 0.0, 1200.0],
        "isFraud": [1, 0, 0],
    })


def test_schema_detects_missing_columns():
    missing = validate_schema(
        ["type", "amount"],
        ["type", "amount", "isFraud"],
    )

    assert missing == ["isFraud"]


def test_valid_data_has_no_critical_findings():
    df = make_transactions()

    findings = inspect_chunk(df, "isFraud", ALLOWED_TYPES)

    assert findings["missing_values"] == 0
    assert findings["invalid_target_rows"] == 0
    assert findings["invalid_transaction_type_rows"] == 0
    assert findings["negative_amount_rows"] == 0


def test_zero_amount_is_counted_but_preserved():
    df = make_transactions()

    findings = inspect_chunk(df, "isFraud", ALLOWED_TYPES)

    assert findings["zero_amount_rows"] == 1
    assert len(df) == 3


def test_invalid_target_requires_review():
    df = make_transactions()
    df.loc[0, "isFraud"] = 2

    findings = inspect_chunk(df, "isFraud", ALLOWED_TYPES)

    assert findings["invalid_target_rows"] == 1
    assert get_quality_status(findings) == "review_required"


def test_negative_amount_requires_review():
    df = make_transactions()
    df.loc[0, "amount"] = -500

    findings = inspect_chunk(df, "isFraud", ALLOWED_TYPES)

    assert findings["negative_amount_rows"] == 1
    assert get_quality_status(findings) == "review_required"


def test_unexpected_transaction_type_requires_review():
    df = make_transactions()
    df.loc[0, "type"] = "UNKNOWN"

    findings = inspect_chunk(df, "isFraud", ALLOWED_TYPES)

    assert findings["invalid_transaction_type_rows"] == 1
    assert get_quality_status(findings) == "review_required"
