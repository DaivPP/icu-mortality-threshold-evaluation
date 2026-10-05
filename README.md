# Revisiting Clinical Severity Prediction: From Rule-Derived Labels to Real Outcomes

Full write-up: [REPORT.md](./REPORT.md)

A follow-up to an earlier capstone project (HEHCSP), which trained a classifier on
synthetic, rule-derived triage labels and reported misleadingly high accuracy. This project
repeats the exercise on real clinical data — in-hospital mortality from the MIMIC-IV
Clinical Database Demo — with proper cross-validation and threshold-aware evaluation under
class imbalance.

**Headline result:** early vital signs carry real but modest predictive signal
(PR-AUC ≈ 0.18–0.19 against an 11.7% base rate). A Random Forest model, once its decision
threshold is chosen fairly, modestly outperforms a transparent, un-fit clinical
threshold rule — a much more honest and interesting finding than a single inflated
accuracy number.

## Reproducing this

```
pip install pandas numpy scikit-learn

# 1. Build the feature table (first-24h vitals + real mortality outcome)
python src/build_features.py

# 2. Baseline comparison at the default 0.5 threshold
#    (demonstrates why this is a misleading comparison under class imbalance)
python src/train_baseline.py

# 3. Fair, threshold-tuned comparison (the real result)
python src/threshold_tuning.py
```

Data: [MIMIC-IV Clinical Database Demo](https://physionet.org/content/mimic-iv-demo/)
(openly accessible, no credentialing required). For any formal submission, obtain the data
directly from PhysioNet rather than the development mirror used here
([TinyEHR](https://github.com/vidulpanickan/TinyEHR)).

## Project structure

```
src/
  build_features.py      # cohort + feature extraction from raw MIMIC-IV tables
  train_baseline.py      # dummy / rule / LR / RF comparison at default threshold
  threshold_tuning.py    # fair comparison with recall-matched thresholds
outputs/
  feature_table.csv
  model_comparison.csv
  threshold_tuned_comparison.csv
  feature_importance.csv
REPORT.md                # full write-up
```
