"""
evaluate.py — day-based test split evaluation.

Uses a held-out *day* (the most recent day in the dataset) as the test set
to avoid leakage from overlapping windows within the same session.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)


def day_based_eval(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    clf,
    test_day: str | None = None,
) -> dict[str, float]:
    """Train on all days except *test_day*, evaluate on *test_day*.

    Args:
        X:        Feature matrix.
        y:        Binary labels (0/1).
        groups:   Day-string per sample (e.g. "2026-07-07").
        clf:      Unfitted sklearn-compatible classifier.
        test_day: Day string to hold out. If None, uses the lexicographically
                  last (most recent) day.

    Returns:
        Dict with keys: auc, f1, precision, recall, test_day, n_train, n_test.
    """
    unique_days = np.unique(groups)
    if test_day is None:
        test_day = unique_days[-1]  # most recent

    train_mask = groups != test_day
    test_mask  = groups == test_day

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    metrics = {
        "test_day":  test_day,
        "n_train":   int(train_mask.sum()),
        "n_test":    int(test_mask.sum()),
        "auc":       float(roc_auc_score(y_test, y_prob)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
    }
    print(f"\nDay-based eval — held-out day: {test_day}")
    print(f"  train n={metrics['n_train']}  test n={metrics['n_test']}")
    print(classification_report(y_test, y_pred, zero_division=0))
    return metrics
