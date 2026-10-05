"""
Fix the unfair comparison from train_baseline.py: the default 0.5 classification
threshold is not meaningful on 88/12 imbalanced data. Here we:

  1. Compute PR-AUC (average precision) -- a threshold-independent summary
     metric that is more appropriate than ROC-AUC for rare-event prediction.
  2. For each model, find the probability threshold that achieves roughly the
     SAME recall as the simple rule baseline (80%), then compare precision/F1
     at that matched operating point. This is the fair comparison: can the
     model catch deaths as well as the rule, while raising fewer false alarms?
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    recall_score, precision_score, f1_score, confusion_matrix
)

from train_baseline import load_data, simple_threshold_rule, RANDOM_STATE, N_SPLITS, N_REPEATS

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
TARGET_RECALL = 0.80  # matches the rule baseline, for a fair head-to-head


def threshold_for_target_recall(y_true, y_proba, target_recall):
    """Find the lowest-precision-cost threshold that achieves >= target_recall."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns arrays 1 longer than thresholds; align them
    precisions, recalls = precisions[:-1], recalls[:-1]
    candidates = np.where(recalls >= target_recall)[0]
    if len(candidates) == 0:
        return 0.0  # can't hit target recall even at threshold 0
    # among thresholds achieving target recall, pick the one with best precision
    best_idx = candidates[np.argmax(precisions[candidates])]
    return thresholds[best_idx]


def evaluate_at_threshold(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fn": cm[1][0],
        "fp": cm[0][1],
        "threshold": threshold,
    }


def run_model_threshold_analysis(name, model_fn, X, y, n_repeats=N_REPEATS):
    pr_aucs = []
    tuned_results = []

    for r in range(n_repeats):
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + r)
        model = model_fn()
        y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

        pr_aucs.append(average_precision_score(y, y_proba))

        thresh = threshold_for_target_recall(y, y_proba, TARGET_RECALL)
        tuned_results.append(evaluate_at_threshold(y, y_proba, thresh))

    tuned_df = pd.DataFrame(tuned_results)
    print(f"\n--- {name}: threshold tuned for ~{TARGET_RECALL:.0%} recall "
          f"(mean over {n_repeats} repeats) ---")
    print(f"PR-AUC (average precision, threshold-independent): "
          f"{np.mean(pr_aucs):.3f} +/- {np.std(pr_aucs):.3f}")
    print(f"At matched recall target:")
    for metric in ["recall", "precision", "f1", "fn", "fp"]:
        print(f"  {metric:>10}: {tuned_df[metric].mean():.3f} +/- {tuned_df[metric].std():.3f}")

    return {
        "model": name,
        "pr_auc": np.mean(pr_aucs),
        "tuned_recall": tuned_df["recall"].mean(),
        "tuned_precision": tuned_df["precision"].mean(),
        "tuned_f1": tuned_df["f1"].mean(),
        "tuned_fn": tuned_df["fn"].mean(),
        "tuned_fp": tuned_df["fp"].mean(),
    }


def main():
    X, y, feature_cols, df = load_data()
    print(f"N = {len(y)}, positive (died) = {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"Target recall for fair comparison: {TARGET_RECALL:.0%} (matches the rule baseline)\n")

    # Rule baseline, for reference in the same table
    y_pred_rule = simple_threshold_rule(X)
    cm = confusion_matrix(y, y_pred_rule, labels=[0, 1])
    rule_row = {
        "model": "Threshold rule (unfit)",
        "pr_auc": np.nan,
        "tuned_recall": recall_score(y, y_pred_rule),
        "tuned_precision": precision_score(y, y_pred_rule, zero_division=0),
        "tuned_f1": f1_score(y, y_pred_rule, zero_division=0),
        "tuned_fn": cm[1][0],
        "tuned_fp": cm[0][1],
    }

    results = [rule_row]

    logreg_fn = lambda: Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
    ])
    results.append(run_model_threshold_analysis("Logistic Regression", logreg_fn, X, y))

    rf_fn = lambda: Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=5, class_weight="balanced", random_state=RANDOM_STATE
        )),
    ])
    results.append(run_model_threshold_analysis("Random Forest", rf_fn, X, y))

    summary = pd.DataFrame(results)
    print("\n=== FAIR COMPARISON: all models/rule matched to ~80% recall ===")
    print(summary.to_string(index=False))
    summary.to_csv(OUT_DIR / "threshold_tuned_comparison.csv", index=False)
    print(f"\nSaved: {OUT_DIR/'threshold_tuned_comparison.csv'}")


if __name__ == "__main__":
    main()
