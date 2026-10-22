
from pathlib import Path
from unittest import result

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"


def main():
    reports = []

    for split in ["train", "validation", "test"]:
        path = DATA_DIR / f"{split}.csv"

        df = pd.read_csv(
            path,
            usecols=["amount", "oldbalanceOrg", "isFraud"],
        )

        y_true = df["isFraud"].astype("int8")
        y_pred = (
            df["amount"] == df["oldbalanceOrg"]
        ).astype("int8")

        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()

        result = {
            "split": split,
            "transactions": len(df),
            "actual_fraud": int(y_true.sum()),
            "flagged": int(y_pred.sum()),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
            "precision": precision_score(
                y_true, y_pred, zero_division=0
            ),
            "recall": recall_score(
                y_true, y_pred, zero_division=0
            ),
            "f1": f1_score(
                y_true, y_pred, zero_division=0
            ),
        }

        reports.append(result)

        print(f"\n{'=' * 60}")
        print(f"{split.upper()} — EXACT MATCH RULE")



        for key, value in result.items():
            if key in {"precision", "recall", "f1"}:
               print(f"{key}: {value:.4%}")
            elif isinstance(value, (int, float)):
               print(f"{key}: {value:,}")
            else:
               print(f"{key}: {value}")





        del df

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "exact_match_rule_benchmark.csv"

    pd.DataFrame(reports).to_csv(output, index=False)
    print(f"\nReport saved to: {output}")


if __name__ == "__main__":
    main()
