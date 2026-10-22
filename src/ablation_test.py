
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

TARGET = "isFraud"

# Remove the two dominant engineered features for this experiment.
REDUCED_FEATURES = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "log_amount",
    "amount_to_destination_balance",
]

NUMERIC = [
    "step",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "log_amount",
    "amount_to_destination_balance",
]
CATEGORICAL = ["type"]


def build_preprocessor():
    return ColumnTransformer([
        ("numeric", "passthrough", NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         CATEGORICAL),
    ])


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")

    # Reproducible stratified sample for a manageable experiment.
    sample, _ = train_test_split(
        train,
        train_size=500_000,
        stratify=train[TARGET],
        random_state=42,
    )

    X_train = sample[REDUCED_FEATURES]
    y_train = sample[TARGET].astype(int)
    X_val = validation[REDUCED_FEATURES]
    y_val = validation[TARGET].astype(int)

    print(f"Training sample: {len(sample):,}")
    print(f"Fraud in sample: {int(y_train.sum()):,}")
    print(f"Validation rows: {len(validation):,}")
    print("\nTraining reduced-feature models...")

    models = {
        "Random Forest reduced": RandomForestClassifier(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
        "XGBoost reduced": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        ),
    }

    results = []

    for name, classifier in models.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", classifier),
        ])

        print(f"\nFitting {name}...")
        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, probabilities)

        results.append({
            "model": name,
            "features_used": len(REDUCED_FEATURES),
            "validation_pr_auc": pr_auc,
        })

        print(f"Validation PR-AUC: {pr_auc:.6f}")

    output = pd.DataFrame(results)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "feature_ablation.csv"
    output.to_csv(path, index=False)

    print("\n=== REDUCED-FEATURE RESULTS ===")
    print(output.to_string(index=False))
    print(f"\nSaved: {path}")
    print(
        "\nCompare these results with the original validation "
        "PR-AUCs: Random Forest 0.9981, XGBoost 0.9866."
    )


if __name__ == "__main__":
    main()
