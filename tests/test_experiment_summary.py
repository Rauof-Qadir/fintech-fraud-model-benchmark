
import json

import pandas as pd
import pytest

import src.build_experiment_summary as summary_module


MODELS = {
    "Logistic Regression": 0.65,
    "Random Forest": 0.20,
    "XGBoost": 0.07,
}


def create_sample_reports(directory):
    """Create small input reports so tests do not need real local reports."""

    for model_name, filename in summary_module.VALIDATION_FILES.items():
        validation = {
            "training_rows": 100,
            "original_training_rows": 100,
            "training_fraud_cases": 5,
            "validation_rows": 20,
            "validation_fraud_cases": 2,
            "threshold": 0.5,
            "average_precision_pr_auc": 0.8,
            "roc_auc": 0.9,
            "precision": 0.7,
            "recall": 0.6,
            "f1": 0.65,
        }

        with (directory / filename).open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(validation, file)

    test_rows = []
    benchmark_rows = []

    for model_name, threshold in MODELS.items():
        test_rows.append({
            "model": model_name,
            "threshold": threshold,
            "pr_auc": 0.81,
            "roc_auc": 0.91,
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.74,
        })

        benchmark_rows.append({
            "model": model_name,
            "training_rows": 100,
            "elapsed_seconds": 1.2,
        })

    pd.DataFrame(test_rows).to_csv(
        directory / "test_evaluation.csv", index=False
    )
    pd.DataFrame(benchmark_rows).to_csv(
        directory / "model_benchmark.csv", index=False
    )


@pytest.fixture
def sample_reports(tmp_path, monkeypatch):
    create_sample_reports(tmp_path)

    monkeypatch.setattr(summary_module, "REPORTS", tmp_path)
    monkeypatch.setattr(
        summary_module, "OUTPUT", tmp_path / "experiment_summary.json"
    )

    return tmp_path


def read_summary(directory):
    with (directory / "experiment_summary.json").open(
        "r", encoding="utf-8"
    ) as file:
        return json.load(file)


def test_summary_contains_all_three_models(sample_reports):
    summary_module.main()

    summary = read_summary(sample_reports)
    models = {item["model"] for item in summary["experiments"]}

    assert models == set(MODELS)


def test_summary_records_expected_thresholds(sample_reports):
    summary_module.main()

    summary = read_summary(sample_reports)
    experiments = {
        item["model"]: item for item in summary["experiments"]
    }

    for model_name, threshold in MODELS.items():
        assert experiments[model_name]["validation_threshold"] == 0.5
        assert experiments[model_name]["test_threshold"] == threshold


def test_summary_includes_reproducibility_limitations(sample_reports):
    summary_module.main()

    summary = read_summary(sample_reports)
    limitations = summary["important_limitations"]

    assert any("synthetic" in item.lower() for item in limitations)
    assert any("threshold" in item.lower() for item in limitations)
    assert summary["split_strategy"] == "Chronological"
    assert summary["split_boundaries"]["test_steps"] == "632-743"


def test_missing_report_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(summary_module, "REPORTS", tmp_path)
    monkeypatch.setattr(
        summary_module, "OUTPUT", tmp_path / "experiment_summary.json"
    )

    with pytest.raises(FileNotFoundError, match="Required report not found"):
        summary_module.main()
