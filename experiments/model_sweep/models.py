"""
models.py - model implementations, each evaluated via day-based
LOGO CV. Plain sklearn classifiers (LR, GaussianNB, RF, GBT) plug directly
into host/training/cross_validation.py:day_based_cv unchanged - it's
generic over any sklearn-compatible classifier. Two need custom CV loops
because day_based_cv has no hook for what they require:

  - nested_elasticnet_day_cv: an INNER day-based hyperparameter search
    inside each OUTER LOGO fold (day_based_cv only does one level).
  - conditional_logit_day_cv: fits on within-day-matched-pair DIFFERENCES,
    not on (X, y) directly - a structurally different training objective.

Both mirror day_based_cv's exact splitting/per-fold-scaler/metric
conventions, just with the one additional step each requires - same
pattern already used in experiments/informed_prior/evaluate.py's
informed_day_based_cv for the same reason.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

# Reduced from a finer grid for tractable runtime given nested CV's cost
# (outer folds x inner folds x grid) - still spans pure-L2 to pure-L1 and
# three orders of magnitude of regularization strength.
ELASTICNET_L1_RATIOS = (0.0, 0.5, 1.0)
ELASTICNET_CS = (0.01, 0.1, 1.0)


def _aggregate(aucs: list[float], f1s: list[float]) -> dict:
    aucs_arr, f1s_arr = np.array(aucs), np.array(f1s)
    return {
        "mean_auc": float(aucs_arr.mean()) if len(aucs_arr) else float("nan"),
        "std_auc":  float(aucs_arr.std())  if len(aucs_arr) else float("nan"),
        "mean_f1":  float(f1s_arr.mean())  if len(f1s_arr) else float("nan"),
        "std_f1":   float(f1s_arr.std())   if len(f1s_arr) else float("nan"),
        "fold_aucs": aucs_arr.tolist(),
        "fold_f1s":  f1s_arr.tolist(),
    }


def simple_classifiers() -> dict[str, object]:
    """Plug directly into host.training.cross_validation.day_based_cv."""
    return {
        "lr_l2_baseline": LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced"),
        "gaussian_nb": GaussianNB(),
        "rf_shallow": RandomForestClassifier(
            n_estimators=200, max_depth=3, min_samples_leaf=8,
            class_weight="balanced", random_state=42),
        "gbt_shallow": GradientBoostingClassifier(
            n_estimators=100, max_depth=2, min_samples_leaf=8,
            learning_rate=0.05, random_state=42),
    }


def nested_elasticnet_day_cv(X, y, groups, sample_weight=None) -> tuple[dict, list]:
    """Outer day-based LOGO; inner day-based LOGO (among the outer-train
    days only) picks l1_ratio/C by mean inner AUC; refit on the full outer
    train with the chosen params, score on the outer test day. Day-grouping
    is respected at both levels - no window ever crosses a day boundary in
    either the inner or outer split."""
    logo = LeaveOneGroupOut()
    aucs, f1s, chosen = [], [], []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        groups_tr = groups[train_idx]
        w_tr = sample_weight[train_idx] if sample_weight is not None else None
        if len(np.unique(y_te)) < 2:
            continue

        inner_days = sorted(set(groups_tr))
        best_score, best_params = -np.inf, (0.5, 0.1)
        if len(inner_days) >= 2:
            for l1_ratio in ELASTICNET_L1_RATIOS:
                for C in ELASTICNET_CS:
                    inner_aucs = []
                    for inner_test_day in inner_days:
                        itr_mask = groups_tr != inner_test_day
                        ite_mask = groups_tr == inner_test_day
                        y_ite = y_tr[ite_mask]
                        if len(np.unique(y_ite)) < 2 or len(np.unique(y_tr[itr_mask])) < 2:
                            continue
                        scaler = StandardScaler()
                        Xi_tr = scaler.fit_transform(X_tr[itr_mask])
                        Xi_te = scaler.transform(X_tr[ite_mask])
                        clf = LogisticRegression(
                            penalty="elasticnet", solver="saga", l1_ratio=l1_ratio, C=C,
                            max_iter=3000, class_weight="balanced")
                        fit_kwargs = {"sample_weight": w_tr[itr_mask]} if w_tr is not None else {}
                        clf.fit(Xi_tr, y_tr[itr_mask], **fit_kwargs)
                        inner_aucs.append(roc_auc_score(y_ite, clf.predict_proba(Xi_te)[:, 1]))
                    if inner_aucs and np.mean(inner_aucs) > best_score:
                        best_score = np.mean(inner_aucs)
                        best_params = (l1_ratio, C)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        clf = LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=best_params[0], C=best_params[1],
            max_iter=3000, class_weight="balanced")
        fit_kwargs = {"sample_weight": w_tr} if w_tr is not None else {}
        clf.fit(X_tr_s, y_tr, **fit_kwargs)
        y_prob = clf.predict_proba(X_te_s)[:, 1]
        y_pred = clf.predict(X_te_s)
        aucs.append(roc_auc_score(y_te, y_prob))
        f1s.append(f1_score(y_te, y_pred, zero_division=0))
        chosen.append(best_params)

    return _aggregate(aucs, f1s), chosen


def match_pairs_by_day(y: np.ndarray, groups: np.ndarray, timestamps: np.ndarray) -> list[tuple[int, int]]:
    """Pair each positive with its temporally-nearest same-day negative.
    Operates on whatever index subset is passed in (caller restricts to a
    train split first) - pairing never looks outside the given arrays, so
    it can't cross a day or a train/test boundary."""
    pairs = []
    for day in sorted(set(groups)):
        day_mask = groups == day
        pos_idx = np.flatnonzero(day_mask & (y == 1))
        neg_idx = np.flatnonzero(day_mask & (y == 0))
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue
        neg_ts = timestamps[neg_idx]
        for pi in pos_idx:
            nearest = neg_idx[np.argmin(np.abs(neg_ts - timestamps[pi]))]
            pairs.append((int(pi), int(nearest)))
    return pairs


def conditional_logit_day_cv(X, y, groups, timestamps) -> dict:
    """1:1 matched-pair conditional logistic regression via the standard
    paired-difference reformulation: for matched (case, control), fit an
    unpenalized, no-intercept logistic regression on +(X_case - X_control)
    labeled 1 and -(X_case - X_control) labeled 0. This is the standard
    equivalent formulation of a fixed-effects/conditional logit for 1:1
    matching (each day acts as its own stratum via the matching itself,
    rather than needing an explicit strata term) - used here because no
    conditional-logit estimator ships in sklearn/statsmodels that plugs in
    as a drop-in classifier with predict_proba for arbitrary features.
    """
    logo = LeaveOneGroupOut()
    aucs, f1s = [], []

    for train_idx, test_idx in logo.split(X, y, groups):
        y_te = y[test_idx]
        if len(np.unique(y_te)) < 2:
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X[train_idx])
        X_te_s = scaler.transform(X[test_idx])

        pairs = match_pairs_by_day(y[train_idx], groups[train_idx], timestamps[train_idx])
        if len(pairs) < 3:
            continue  # not enough matched pairs in this fold's training days to fit anything meaningful

        diffs, labels = [], []
        for pi, ni in pairs:
            d = X_tr_s[pi] - X_tr_s[ni]
            diffs.append(d); labels.append(1)
            diffs.append(-d); labels.append(0)

        clf = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000)
        clf.fit(np.array(diffs), np.array(labels))

        scores = X_te_s @ clf.coef_[0]
        probs = 1.0 / (1.0 + np.exp(-scores))
        preds = (probs >= 0.5).astype(int)
        aucs.append(roc_auc_score(y_te, probs))
        f1s.append(f1_score(y_te, preds, zero_division=0))

    return _aggregate(aucs, f1s)
