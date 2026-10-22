
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
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

MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"
REPORT_PATH = REPORT_DIR / "logistic_regression_validation.json"

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
TARGET = "isFraud"

THRESHOLD = 0.5


def main():
    start_time = time.time()

    for path in [TRAIN_PATH, VALIDATION_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required prepared dataset not found: {path}\n"
                "Run python -m src.prepare_data first."
            )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training data...")
    train = pd.read_csv(TRAIN_PATH)

    print("Loading validation data...")
    validation = pd.read_csv(VALIDATION_PATH)

    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    for name, frame in [
        ("training", train),
        ("validation", validation),
    ]:
        missing = set(feature_columns + [TARGET]) - set(frame.columns)
        if missing:
            raise ValueError(
                f"{name} data is missing columns: {sorted(missing)}"
            )

        if frame[TARGET].isna().any():
            raise ValueError(f"{name} target contains missing values.")

        if not frame[TARGET].isin([0, 1]).all():
            raise ValueError(f"{name} target must contain only 0 and 1.")

    X_train = train[feature_columns]
    y_train = train[TARGET].astype("int8")

    X_validation = validation[feature_columns]
    y_validation = validation[TARGET].astype("int8")

    if y_train.nunique() != 2:
        raise ValueError("Training data must contain both target classes.")

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
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
                LogisticRegression(
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=200,
                    random_state=42,
                ),
            ),
        ]
    )

    print("\nTraining Logistic Regression...")
    print(f"Training rows: {len(train):,}")
    print(f"Training fraud cases: {int(y_train.sum()):,}")

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
        "model": "Logistic Regression",
        "threshold": THRESHOLD,
        "model_config": {
         "class_weight": "balanced",
         "solver": "lbfgs",
         "max_iter": 200,
         "random_state": 42,
         "numeric_scaling": "StandardScaler",
         "categorical_encoding": "OneHotEncoder",
},
        "training_rows": int(len(train)),
        "training_fraud_cases": int(y_train.sum()),
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
        "elapsed_seconds": round(time.time() - start_time, 2),
        "evaluation_note": (
            "Validation results only. Threshold 0.5 is a baseline; "
            "the final test set has not been evaluated."
        ),
    }

    joblib.dump(model, MODEL_PATH)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\n=== LOGISTIC REGRESSION: VALIDATION RESULTS ===")
    print(f"PR-AUC / Average Precision: {metrics['average_precision_pr_auc']:.6f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall: {metrics['recall']:.6f}")
    print(f"F1: {metrics['f1']:.6f}")

    print("\nConfusion matrix:")
    print(f"True negatives (TN): {tn:,}")
    print(f"False positives (FP): {fp:,}")
    print(f"False negatives (FN): {fn:,}")
    print(f"True positives (TP): {tp:,}")

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Metrics saved: {REPORT_PATH}")
    print(f"Elapsed seconds: {metrics['elapsed_seconds']}")

    return metrics


if __name__ == "__main__":
    main()
