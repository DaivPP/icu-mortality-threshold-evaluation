"""
Train and evaluate baseline models for early mortality prediction.

Design choices made deliberately to avoid the capstone's core flaw:
  - Label is REAL (hospital_expire_flag), not derived from the input features.
  - Imputation/scaling are fit INSIDE each CV fold (via sklearn Pipeline), so
    no information from validation folds leaks into training.
  - Repeated stratified k-fold CV is used because N=128 is small; a single
    80/20 split would give a noisy, unreliable estimate.
  - A transparent threshold-rule baseline is included so we can honestly ask:
    does the ML model beat a simple, clinically interpretable rule?
  - Headline metric is RECALL on the mortality class, not accuracy --
    accuracy is a misleading metric on an 88/12 imbalanced task.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    recall_score, precision_score, f1_score, roc_auc_score,
    brier_score_loss, confusion_matrix, classification_report
)

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
FEATURE_TABLE = OUT_DIR / "feature_table.csv"

LABEL_COL = "hospital_expire_flag"
VITAL_PREFIXES = ["heart_rate", "sbp", "dbp", "resp_rate", "spo2", "temp_c"]

N_SPLITS = 5
N_REPEATS = 10  # repeated CV to stabilize estimates given small N
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(FEATURE_TABLE)
    feature_cols = [c for c in df.columns if any(c.startswith(p) for p in VITAL_PREFIXES)]
    feature_cols += ["anchor_age"]
    X = df[feature_cols].copy()
    y = df[LABEL_COL].astype(int).values
    return X, y, feature_cols, df


def simple_threshold_rule(X):
    """
    A transparent, interpretable baseline modeled on real early-warning scores
    (loosely inspired by qSOFA / NEWS-style thresholds), NOT fit to this data.
    Flags a patient if >=2 of these are true in their first-24h vitals:
      - min SpO2 < 92
      - min systolic BP < 90
      - max respiratory rate >= 22
      - max heart rate > 110
    This plays the same role the capstone's rule-based system did -- but here
    it is explicitly a BASELINE to beat, not the source of the labels.
    """
    flags = pd.DataFrame(index=X.index)
    flags["low_spo2"] = X["spo2_min"] < 92
    flags["low_sbp"] = X["sbp_min"] < 90
    flags["high_rr"] = X["resp_rate_max"] >= 22
    flags["high_hr"] = X["heart_rate_max"] > 110
    score = flags.sum(axis=1)
    return (score >= 2).astype(int).values


def evaluate_once(y_true, y_pred, y_proba=None):
    """Compute metrics for a single run (one seed's worth of pooled out-of-fold predictions)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan,
        "fn": cm[1][0],
        "tp": cm[1][1],
    }


def repeated_cv_eval(name, model_fn, X, y, n_splits=N_SPLITS, n_repeats=N_REPEATS, needs_proba=True):
    """Run cross_val_predict independently across n_repeats different stratified
    splits (different random seeds), then average the resulting metrics.
    This is the correct way to stabilize an estimate on a small dataset,
    since a single 5-fold split on N=128 is noisy."""
    per_repeat = []
    for r in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE + r)
        model = model_fn()
        y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")
        y_proba = None
        if needs_proba:
            y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        per_repeat.append(evaluate_once(y, y_pred, y_proba))

    df = pd.DataFrame(per_repeat)
    means = df.mean(numeric_only=True)
    stds = df.std(numeric_only=True)

    print(f"\n--- {name} (mean +/- std over {n_repeats} repeats of {n_splits}-fold CV) ---")
    for metric in ["recall", "precision", "f1", "roc_auc"]:
        print(f"{metric:>10}: {means[metric]:.3f} +/- {stds[metric]:.3f}")
    print(f"avg false negatives (missed deaths) per repeat: {means['fn']:.2f} "
          f"out of {y.sum()} actual deaths")

    return {"model": name, **means.to_dict()}


def main():
    X, y, feature_cols, df = load_data()
    print(f"N = {len(y)}, positive (died) = {y.sum()} ({y.mean()*100:.1f}%)")
    print("NOTE: with only 128 samples and 15 positives, treat all metrics as")
    print("rough estimates with wide uncertainty, not precise performance figures.\n")

    results = []

    # --- Baseline 0: majority-class dummy classifier ---
    dummy_fn = lambda: Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", DummyClassifier(strategy="most_frequent")),
    ])
    results.append(repeated_cv_eval("Dummy (predict no-death always)", dummy_fn, X, y, needs_proba=False))

    # --- Baseline 1: transparent threshold rule (not fit to data, no CV needed) ---
    y_pred_rule = simple_threshold_rule(X)
    rule_metrics = evaluate_once(y, y_pred_rule, None)
    print(f"\n--- Threshold rule (qSOFA-style, unfit -- evaluated once, no CV needed) ---")
    for metric in ["recall", "precision", "f1"]:
        print(f"{metric:>10}: {rule_metrics[metric]:.3f}")
    print(f"false negatives (missed deaths): {rule_metrics['fn']} out of {y.sum()} actual deaths")
    results.append({"model": "Threshold rule (unfit)", **rule_metrics})

    # --- Model 1: Logistic Regression ---
    logreg_fn = lambda: Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
    ])
    results.append(repeated_cv_eval("Logistic Regression (balanced)", logreg_fn, X, y))

    # --- Model 2: Random Forest ---
    rf_fn = lambda: Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=5, class_weight="balanced",
            random_state=RANDOM_STATE
        )),
    ])
    results.append(repeated_cv_eval("Random Forest (balanced)", rf_fn, X, y))

    # --- Feature importance (fit once on full data, for interpretation only --
    #     NOT a performance estimate, since it's fit and "evaluated" on the same data) ---
    rf_full = rf_fn()
    rf_full.fit(X, y)
    importances = pd.Series(rf_full.named_steps["clf"].feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    print("\n--- Random Forest feature importance (fit on full data, for interpretation only) ---")
    print(importances.head(10).to_string())

    # --- Summary table ---
    summary = pd.DataFrame(results)
    print("\n=== SUMMARY (repeated 5-fold CV, averaged via pooled out-of-fold predictions) ===")
    print(summary.to_string(index=False))

    summary.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    importances.to_csv(OUT_DIR / "feature_importance.csv")
    print(f"\nSaved: {OUT_DIR/'model_comparison.csv'}, {OUT_DIR/'feature_importance.csv'}")


if __name__ == "__main__":
    main()
