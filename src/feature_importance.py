
from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

FEATURES = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "log_amount",
    "amount_to_origin_balance",
    "amount_to_destination_balance",
    "origin_balance_covers_amount",
]

MODELS = {
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
}


def main():
    validation = pd.read_csv(DATA_DIR / "validation.csv")

    # Sample validation rows to keep this audit manageable.
    sample_size = min(30000, len(validation))
    sample = validation.sample(
        n=sample_size,
        random_state=42,
    )

    X = sample[FEATURES]
    y = sample["isFraud"].astype(int)

    print(f"Validation sample rows: {len(sample):,}")
    print(f"Fraud cases in sample: {int(y.sum()):,}")

    all_results = []

    for model_name, filename in MODELS.items():
        print(f"\nRunning permutation importance: {model_name}")

        model = joblib.load(MODEL_DIR / filename)

        baseline_pr_auc = average_precision_score(
            y, model.predict_proba(X)[:, 1]
        )

        result = permutation_importance(
            model,
            X,
            y,
            scoring="average_precision",
            n_repeats=3,
            random_state=42,
            n_jobs=-1,
        )

        model_results = pd.DataFrame({
            "model": model_name,
            "feature": FEATURES,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }).sort_values("importance_mean", ascending=False)

        all_results.append(model_results)

        print(f"Baseline PR-AUC: {baseline_pr_auc:.6f}")
        print(model_results.to_string(
            index=False,
            formatters={
                "importance_mean": lambda x: f"{x:.6f}",
                "importance_std": lambda x: f"{x:.6f}",
            },
        ))

    report = pd.concat(all_results, ignore_index=True)
    output = REPORT_DIR / "feature_importance.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)

    print(f"\nSaved: {output}")
    print(
        "\nNote: importance shows model dependence, not proof "
        "of causation or proof that leakage is absent."
    )


if __name__ == "__main__":
    main()
