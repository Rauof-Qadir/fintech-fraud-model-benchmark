from pathlib import Path
import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

TARGET = "isFraud"
SAMPLE_SIZE = 500_000
RANDOM_STATE = 42

FEATURE_SETS = {
    "full": [
        "step", "type", "amount", "oldbalanceOrg", "oldbalanceDest",
        "log_amount", "amount_to_origin_balance",
        "amount_to_destination_balance", "origin_balance_covers_amount",
    ],
    "no_origin_balance": [
        "step", "type", "amount", "oldbalanceDest", "log_amount",
        "amount_to_destination_balance",
    ],
    "no_destination_balance": [
        "step", "type", "amount", "oldbalanceOrg", "log_amount",
        "amount_to_origin_balance", "origin_balance_covers_amount",
    ],
    "no_step": [
        "type", "amount", "oldbalanceOrg", "oldbalanceDest",
        "log_amount", "amount_to_origin_balance",
        "amount_to_destination_balance", "origin_balance_covers_amount",
    ],
    "core_only": ["type", "amount", "log_amount"],
}


def build_model(model_name, features):
    categorical = [c for c in features if c == "type"]
    numeric = [c for c in features if c != "type"]

    preprocessor = ColumnTransformer([
        ("numeric", "passthrough", numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])

    if model_name == "Random Forest":
        classifier = RandomForestClassifier(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    else:
        classifier = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", classifier),
    ])


def main():
    train_path = DATA_DIR / "train.csv"
    val_path = DATA_DIR / "validation.csv"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            "Train/validation files missing. Run data preparation first."
        )

    print("Loading train and validation data...")
    train = pd.read_csv(train_path)
    validation = pd.read_csv(val_path)

    # Match the reproducible sampling approach in the original model scripts.
    if len(train) > SAMPLE_SIZE:
        sample = train.groupby(
            TARGET, group_keys=False
        ).sample(
            frac=SAMPLE_SIZE / len(train),
            random_state=RANDOM_STATE,
        )
        if len(sample) > SAMPLE_SIZE:
            sample = sample.sample(
                n=SAMPLE_SIZE, random_state=RANDOM_STATE
            )
    else:
        sample = train

    y_train = sample[TARGET].astype("int8")
    y_val = validation[TARGET].astype("int8")

    print(f"Training sample: {len(sample):,}")
    print(f"Training fraud cases: {int(y_train.sum()):,}")
    print(f"Validation rows: {len(validation):,}")
    print("Validation remains chronological and is not resampled.")

    results = []

    for model_name in ["Random Forest", "XGBoost"]:
        for variant, features in FEATURE_SETS.items():
            start = time.time()
            print(f"\nTraining {model_name} | {variant}", flush=True)

            model = build_model(model_name, features)
            model.fit(sample[features], y_train)

            probabilities = model.predict_proba(
                validation[features]
            )[:, 1]

            row = {
                "model": model_name,
                "feature_set": variant,
                "feature_count": len(features),
                "validation_pr_auc": average_precision_score(
                    y_val, probabilities
                ),
                "validation_roc_auc": roc_auc_score(
                    y_val, probabilities
                ),
                "training_rows": len(sample),
                "training_fraud": int(y_train.sum()),
                "validation_rows": len(validation),
                "validation_fraud": int(y_val.sum()),
                "duration_seconds": round(time.time() - start, 1),
            }
            results.append(row)

            print(
                f"PR-AUC={row['validation_pr_auc']:.6f} | "
                f"ROC-AUC={row['validation_roc_auc']:.6f} | "
                f"seconds={row['duration_seconds']}",
                flush=True,
            )

            # Save after every run so partial progress is retained.
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(results).to_csv(
                REPORT_DIR / "controlled_ablation.csv", index=False
            )

    output = pd.DataFrame(results)
    print("\n=== CONTROLLED ABLATION RESULTS ===")
    print(
        output[[
            "model", "feature_set", "validation_pr_auc",
            "validation_roc_auc", "duration_seconds",
        ]].to_string(index=False)
    )
    print(f"\nSaved: {REPORT_DIR / 'controlled_ablation.csv'}")


if __name__ == "__main__":
    main()
