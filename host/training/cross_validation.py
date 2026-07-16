"""
cross_validation.py — day-based leave-one-group-out cross-validation.

Each "group" is a calendar day. Leaves one day out as the test fold per
iteration, avoiding leakage from overlapping windows within a session.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, f1_score


def day_based_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    clf,
) -> dict[str, float | list]:
    """Run LeaveOneGroupOut CV grouped by day.

    Args:
        X:      Feature matrix (n_samples × n_features).
        y:      Binary labels.
        groups: Day-string per sample.
        clf:    Unfitted sklearn-compatible classifier with predict_proba.

    Returns:
        Dict with mean_auc, std_auc, mean_f1, std_f1, fold_aucs, fold_f1s.
    """
    logo   = LeaveOneGroupOut()
    aucs   = []
    f1s    = []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        clf_copy = _clone(clf)
        clf_copy.fit(X_tr, y_tr)

        y_prob = clf_copy.predict_proba(X_te)[:, 1]
        y_pred = clf_copy.predict(X_te)

        if len(np.unique(y_te)) < 2:
            # Can't compute AUC with only one class in test fold — skip
            continue

        aucs.append(roc_auc_score(y_te, y_prob))
        f1s.append(f1_score(y_te, y_pred, zero_division=0))

    aucs = np.array(aucs)
    f1s  = np.array(f1s)

    return {
        "mean_auc":  float(aucs.mean()) if len(aucs) else float("nan"),
        "std_auc":   float(aucs.std())  if len(aucs) else float("nan"),
        "mean_f1":   float(f1s.mean())  if len(f1s) else float("nan"),
        "std_f1":    float(f1s.std())   if len(f1s) else float("nan"),
        "fold_aucs": aucs.tolist(),
        "fold_f1s":  f1s.tolist(),
    }


def _clone(clf):
    """Shallow clone via sklearn if available, otherwise call the class directly."""
    try:
        from sklearn.base import clone
        return clone(clf)
    except Exception:
        return clf.__class__(**clf.get_params())
