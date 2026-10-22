
import numpy as np
import pandas as pd

from src.prepare_data import engineer_features


def sample_transactions():
    return pd.DataFrame({
        "step": [1, 2],
        "type": ["TRANSFER", "PAYMENT"],
        "amount": [500.0, 0.0],
        "oldbalanceOrg": [1000.0, 0.0],
        "oldbalanceDest": [200.0, 0.0],
        "isFraud": [1, 0],
    })


def test_feature_engineering_preserves_row_count():
    df = sample_transactions()

    result = engineer_features(df)

    assert len(result) == len(df)


def test_log_amount_is_calculated_correctly():
    result = engineer_features(sample_transactions())

    assert np.isclose(result.loc[0, "log_amount"], np.log1p(500))
    assert result.loc[1, "log_amount"] == 0


def test_balance_ratios_are_calculated_correctly():
    result = engineer_features(sample_transactions())

    assert np.isclose(
        result.loc[0, "amount_to_origin_balance"],
        500 / 1001,
    )

    assert np.isclose(
        result.loc[0, "amount_to_destination_balance"],
        500 / 201,
    )


def test_origin_balance_coverage_feature():
    result = engineer_features(sample_transactions())

    assert result.loc[0, "origin_balance_covers_amount"] == 1
    assert result.loc[1, "origin_balance_covers_amount"] == 1


def test_post_transaction_balances_and_identifiers_are_excluded():
    result = engineer_features(sample_transactions())

    forbidden_columns = {
        "nameOrig",
        "nameDest",
        "newbalanceOrg",
        "newbalanceDest",
        "isFlaggedFraud",
    }

    assert forbidden_columns.isdisjoint(result.columns)
    assert "isFraud" in result.columns


def test_feature_values_are_finite():
    result = engineer_features(sample_transactions())

    numeric_features = result.select_dtypes(include="number")

    assert np.isfinite(numeric_features.to_numpy()).all()

