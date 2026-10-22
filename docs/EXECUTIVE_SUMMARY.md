# Executive Summary ? FinTech Fraud Model Benchmark

## Project goal

Benchmark fraud-classification models and compare threshold decisions under
explicit, illustrative business-cost assumptions.

## Dataset and split

- Dataset: PaySim-style synthetic transaction data
- Transactions: 6,362,620
- Fraud cases: 8,213 (approximately 0.1291%)
- Train: 6,082,007 rows; 5,781 fraud cases
- Validation: 191,147 rows; 1,180 fraud cases
- Test: 89,466 rows; 1,252 fraud cases

The fraud rate changes from approximately 0.0951% in training to 0.6173% in
validation and 1.3994% in test, indicating a substantial temporal distribution
shift.

## Model benchmark

Previously recorded results on the chronological test split:

| Model | Threshold | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.65 | 0.7805 | 23.33% | 99.36% | 0.3778 |
| Random Forest | 0.20 | 0.9973 | 98.80% | 99.04% | 0.9892 |
| XGBoost | 0.07 | 0.9931 | 95.18% | 99.36% | 0.9723 |

These results are specific to this synthetic dataset and current feature design.
They must not be interpreted as expected performance on real banking data.

## Cost-aware evaluation

The project compares models under 16 combinations of false-positive costs
(5, 25, 50, 100 units) and missed-fraud loss multipliers (10%, 25%, 50%, 100%).
Thresholds are selected using validation data before applying them to test data.

Cost units are illustrative, not verified USD, PKR, or actual bank losses.
The preferred strategy changes with the assumptions; there is no universally
best strategy without a validated operational cost model.

The consolidated local evaluation is available at
`reports/final_cost_aware_model_audit.csv`.

## Feature dependency and risk

Ablation experiments show a large reduction in validation PR-AUC when the
origin-balance feature group is removed. This suggests the model relies heavily
on those features in the synthetic data. It does not by itself prove target
leakage, but feature availability at decision time and synthetic-generator
artifacts require further validation.

The test period has already been examined in multiple exploratory analyses.
Consequently, it is not an untouched final holdout. Stronger claims require
independent future-period data and prospective evaluation.

## Verification

The latest reported local automated test run passed 16 tests.

## Conclusion

This portfolio project demonstrates an end-to-end fraud-model benchmarking
workflow: chronological splitting, feature engineering, model comparison,
threshold selection, cost sensitivity, ablation, and automated testing.

It is an educational research project, not a production-ready fraud prevention
system. Real deployment would require independent data validation, operational
cost calibration, monitoring, security controls, and ongoing drift evaluation.
