from pathlib import Path
import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

TARGET = "isFraud"
RANDOM_STATE = 42
SAMPLE_SIZE = 500_000

BASE = [
    "step", "type", "amount", "oldbalanceOrg", "oldbalanceDest",
    "log_amount", "amount_to_origin_balance",
    "amount_to_destination_balance", "origin_balance_covers_amount",
]

VARIANTS = {
    "full": BASE,
    "without_raw_origin_balance": [
        c for c in BASE if c != "oldbalanceOrg"
    ],
    "without_origin_ratio": [
        c for c in BASE if c != "amount_to_origin_balance"
    ],
    "without_origin_coverage_flag": [
        c for c in BASE if c != "origin_balance_covers_amount"
    ],
    "origin_signal_only": [
        "type", "amount", "oldbalanceOrg",
        "amount_to_origin_balance", "origin_balance_covers_amount",
    ],
}


def make_model(name, features):
    numeric = [c for c in features if c != "type"]
    preprocessor = ColumnTransformer([
        ("num", "passthrough", numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["type"]),
    ])

    if name == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    else:
        clf = XGBClassifier(
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

    return Pipeline([("prep", preprocessor), ("model", clf)])


def main():
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "validation.csv")

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
    y_val = val[TARGET].astype("int8")
    results = []

    for model_name in ["Random Forest", "XGBoost"]:
        for variant, features in VARIANTS.items():
            start = time.time()
            print(f"\n{model_name} | {variant}", flush=True)

            model = make_model(model_name, features)
            model.fit(sample[features], y_train)
            prob = model.predict_proba(val[features])[:, 1]

            result = {
                "model": model_name,
                "variant": variant,
                "feature_count": len(features),
                "validation_pr_auc": average_precision_score(y_val, prob),
                "seconds": round(time.time() - start, 1),
            }
            results.append(result)

            REPORTS.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(results).to_csv(
                REPORTS / "origin_feature_ablation.csv", index=False
            )
            print(result, flush=True)

    print("\n=== ORIGIN FEATURE ABLATION ===")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
