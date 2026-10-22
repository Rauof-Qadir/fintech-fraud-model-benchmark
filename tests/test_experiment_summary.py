
import json

import pandas as pd
import pytest

from src.build_experiment_summary import main


def test_summary_contains_all_three_models():
    main()

    path = "reports/experiment_summary.json"

    with open(path, "r", encoding="utf-8") as file:
        summary = json.load(file)

    models = {item["model"] for item in summary["experiments"]}

    assert models == {
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
    }


def test_summary_records_expected_thresholds():
    main()

    with open(
        "reports/experiment_summary.json",
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(file)

    experiments = {
        item["model"]: item
        for item in summary["experiments"]
    }

    assert experiments["Logistic Regression"]["validation_threshold"] == 0.5
    assert experiments["Random Forest"]["validation_threshold"] == 0.5
    assert experiments["XGBoost"]["validation_threshold"] == 0.5

    assert experiments["Logistic Regression"]["test_threshold"] == 0.65
    assert experiments["Random Forest"]["test_threshold"] == 0.20
    assert experiments["XGBoost"]["test_threshold"] == 0.07


def test_summary_includes_reproducibility_limitations():
    main()

    with open(
        "reports/experiment_summary.json",
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(file)

    limitations = summary["important_limitations"]

    assert any("synthetic" in item.lower() for item in limitations)
    assert any("threshold" in item.lower() for item in limitations)
    assert summary["split_strategy"] == "Chronological"


def test_missing_report_raises_clear_error(tmp_path, monkeypatch):
    import src.build_experiment_summary as summary_module

    monkeypatch.setattr(summary_module, "REPORTS", tmp_path)

    with pytest.raises(FileNotFoundError, match="Required report not found"):
        main()
