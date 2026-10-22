
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"

MODEL_PATH = MODEL_DIR / "random_forest.joblib"
REPORT_PATH = REPORT_DIR / "random_forest_validation.json"

RANDOM_STATE = 42
SAMPLE_SIZE = 500_000
THRESHOLD = 0.5

NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "log_amount",
    "amount_to_origin_balance",
    "amount_to_destination_balance",
    "origin_balance_covers_amount",
]

CATEGORICAL_FEATURES = ["type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "isFraud"


def main():
    start = time.time()

    if not TRAIN_PATH.exists() or not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            "Prepared train/validation files missing. "
            "Run python -m src.prepare_data first."
        )

    print("Loading training data...")
    train = pd.read_csv(TRAIN_PATH)

    print("Loading validation data...")
    validation = pd.read_csv(VALIDATION_PATH)

    # Stratified sampling retains both target classes in roughly
    # the same proportion as the original training set.
    if len(train) > SAMPLE_SIZE:
        train_sample = train.groupby(
            TARGET, group_keys=False
        ).sample(
            n=None,
            frac=SAMPLE_SIZE / len(train),
            random_state=RANDOM_STATE,
        )

        # Keep sample size bounded and reproducible.
        if len(train_sample) > SAMPLE_SIZE:
            train_sample = train_sample.sample(
                n=SAMPLE_SIZE,
                random_state=RANDOM_STATE,
            )
    else:
        train_sample = train

    X_train = train_sample[FEATURES]
    y_train = train_sample[TARGET].astype("int8")

    X_validation = validation[FEATURES]
    y_validation = validation[TARGET].astype("int8")

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=18,
                    min_samples_leaf=10,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    print("\nTraining Random Forest...")
    print(f"Original training rows: {len(train):,}")
    print(f"Sampled training rows: {len(train_sample):,}")
    print(f"Sampled fraud cases: {int(y_train.sum()):,}")
    print(f"Validation rows: {len(validation):,}")

    model.fit(X_train, y_train)

    print("\nScoring validation data...")
    probabilities = model.predict_proba(X_validation)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype("int8")

    tn, fp, fn, tp = confusion_matrix(
        y_validation,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "model": "Random Forest",
        "threshold": THRESHOLD,
        "original_training_rows": int(len(train)),
        "sampled_training_rows": int(len(train_sample)),
        "sampled_training_fraud_cases": int(y_train.sum()),
        "validation_rows": int(len(validation)),
        "validation_fraud_cases": int(y_validation.sum()),
        "average_precision_pr_auc": float(
            average_precision_score(y_validation, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y_validation, probabilities)
        ),
        "precision": float(
            precision_score(
                y_validation, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_validation, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_validation, predictions, zero_division=0
            )
        ),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "elapsed_seconds": round(time.time() - start, 2),
        "note": (
            "Random Forest trained on a stratified sample. "
            "Threshold 0.5 is a baseline. Validation results only."
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\n=== RANDOM FOREST: VALIDATION RESULTS ===")
    print(f"PR-AUC / Average Precision: {metrics['average_precision_pr_auc']:.6f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall: {metrics['recall']:.6f}")
    print(f"F1: {metrics['f1']:.6f}")

    print("\nConfusion matrix:")
    print(f"TN: {tn:,}")
    print(f"FP: {fp:,}")
    print(f"FN: {fn:,}")
    print(f"TP: {tp:,}")

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Metrics saved: {REPORT_PATH}")
    print(f"Elapsed seconds: {metrics['elapsed_seconds']}")

    return metrics


if __name__ == "__main__":
    main()
