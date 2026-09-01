"""
thermal_experiment.py - does adding the thermal channel (ambient_c,
object_c) to the feature set change predictive performance?

Pre-registered plan (written before looking at results):
  1. Build the standard window set once via host.training.build_dataset's
     collect_windows() (buffer_ms=7000, x_window_ms=60000 - current
     pipeline defaults, unmodified). Both feature variants below are
     extracted from the SAME window objects, so this is a matched-pairs
     comparison (identical rows under both conditions) - no
     cross-population caveat needed, unlike the buffer-ms sweep.
  2. Variant "no_thermal": host.pipeline.features.compute_features(),
     exactly as the real pipeline uses it today.
  3. Variant "with_thermal": compute_features() + 12 new features (mean,
     std, min, max, rms, zcr, trend for ambient_c and object_c - trend =
     last-sample minus first-sample, a cheap slope proxy). ambient_c/
     object_c are already present in every raw packet (see packet.h) but
     were never wired into compute_features() - see host/pipeline/
     features.py's docstring, which only lists accel/gyro/ppg_ir.
  4. Score both with day-based LOGO CV using GradientBoostingClassifier
     (n_estimators=200, max_depth=3, random_state=42) - the actual best
     model in host/training/train.py today, not a synthetic baseline.
  5. Paired day-block bootstrap (same day-draws applied to both variants)
     to see whether any difference clears noise.

host/pipeline/, host/training/ are imported read-only. Nothing there is
modified.

Usage:
    python -m experiments.thermal_and_buffer.thermal_experiment
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from host.pipeline.features import _time_domain, compute_features
from host.training.build_dataset import (
    DEFAULT_X_WINDOW_MS,
    _clean_nans,
    _min_samples_for_window,
    collect_windows,
)
from host.training.cross_validation import day_based_cv

OUT_PATH = Path(__file__).parent / "thermal_results.json"
N_BOOTSTRAP = 200
SEED = 42
THERMAL_COLS = ("ambient_c", "object_c")


def compute_features_with_thermal(window_df: pd.DataFrame) -> dict[str, float]:
    feats = compute_features(window_df)
    for col in THERMAL_COLS:
        sig = window_df[col].to_numpy(dtype=float)
        feats.update(_time_domain(sig, col))
        feats[f"{col}_trend"] = float(sig[-1] - sig[0]) if len(sig) > 1 else 0.0
    return feats


def _extract(windows, feature_fn, min_samples):
    rows, labels, groups = [], [], []
    feature_names = None
    for w in windows:
        if len(w.df) < min_samples:
            continue
        feats = feature_fn(w.df)
        if feature_names is None:
            feature_names = sorted(feats.keys())
        rows.append([feats[k] for k in feature_names])
        labels.append(w.label)
        groups.append(w.day_str)
    if not rows:
        raise RuntimeError("No windows produced.")
    X = np.stack(rows)
    y = np.array(labels, dtype=int)
    groups_arr = np.array(groups)
    X, feature_names = _clean_nans(X, feature_names)
    return X, y, groups_arr, feature_names


def _gbt_fit_predict(X_tr, y_tr, X_te):
    scaler = StandardScaler()
    X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
    clf.fit(X_tr_s, y_tr)
    return clf.predict_proba(X_te_s)[:, 1]


def _lr_fit_predict(X_tr, y_tr, X_te):
    # Same anchor-reproducing LR config used throughout experiments/ to
    # stay comparable with the documented 0.596 +/- 0.104 baseline.
    scaler = StandardScaler()
    X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)
    clf = LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced")
    clf.fit(X_tr_s, y_tr)
    return clf.predict_proba(X_te_s)[:, 1]


def replicate_aucs_for_draws(X, y, groups, fit_predict_fn, day_draws) -> np.ndarray:
    """Same algorithm as experiments/informed_prior/evaluate.py's
    day_block_bootstrap, duplicated here rather than imported so this
    experiment stays self-contained."""
    out = np.full(len(day_draws), np.nan)
    for i, sample in enumerate(day_draws):
        unique_in_sample = sorted(set(sample))
        if len(unique_in_sample) < 2:
            continue
        fold_aucs = []
        for test_day in unique_in_sample:
            train_days = [d for d in sample if d != test_day]
            if not train_days:
                continue
            train_idx = np.concatenate([np.flatnonzero(groups == d) for d in train_days])
            test_idx = np.flatnonzero(groups == test_day)
            y_tr, y_te = y[train_idx], y[test_idx]
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue
            y_prob = fit_predict_fn(X[train_idx], y_tr, X[test_idx])
            fold_aucs.append(roc_auc_score(y_te, y_prob))
        if fold_aucs:
            out[i] = float(np.mean(fold_aucs))
    return out


def main():
    print("Building window set (buffer_ms=7000, x_window_ms=60000, current defaults) ...")
    windows = collect_windows()
    min_samples = _min_samples_for_window(DEFAULT_X_WINDOW_MS)

    print("\nExtracting no_thermal features (compute_features(), unchanged) ...")
    X_base, y, groups, names_base = _extract(windows, compute_features, min_samples)
    print(f"  X={X_base.shape}  positive_rate={y.mean():.2%}  days={sorted(set(groups))}")

    print("\nExtracting with_thermal features (compute_features() + ambient_c/object_c stats) ...")
    X_therm, y2, groups2, names_therm = _extract(windows, compute_features_with_thermal, min_samples)
    print(f"  X={X_therm.shape}  positive_rate={y2.mean():.2%}")
    assert np.array_equal(y, y2) and np.array_equal(groups, groups2), (
        "Row order diverged between the two feature extractions - should be "
        "impossible since both iterate the same `windows` list in the same "
        "order, but asserting rather than silently trusting it."
    )
    new_thermal_features = [n for n in names_therm if n not in set(names_base)]
    print(f"  {len(new_thermal_features)} new thermal features: {new_thermal_features}")

    print("\n--- Day-based LOGO CV (GBT, the current best model in host/training/train.py) ---")
    cv_base_gbt = day_based_cv(X_base, y, groups, GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42))
    cv_therm_gbt = day_based_cv(X_therm, y, groups, GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42))
    print(f"  no_thermal   AUC={cv_base_gbt['mean_auc']:.3f} +/- {cv_base_gbt['std_auc']:.3f}")
    print(f"  with_thermal AUC={cv_therm_gbt['mean_auc']:.3f} +/- {cv_therm_gbt['std_auc']:.3f}")

    print("\n--- Day-based LOGO CV (LR C=0.01, anchor-reproducing config used elsewhere in experiments/) ---")
    cv_base_lr = day_based_cv(X_base, y, groups, LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced"))
    cv_therm_lr = day_based_cv(X_therm, y, groups, LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced"))
    print(f"  no_thermal   AUC={cv_base_lr['mean_auc']:.3f} +/- {cv_base_lr['std_auc']:.3f}")
    print(f"  with_thermal AUC={cv_therm_lr['mean_auc']:.3f} +/- {cv_therm_lr['std_auc']:.3f}")

    print(f"\n--- Paired day-block bootstrap (N={N_BOOTSTRAP}, matched population - same rows both sides) ---")
    unique_days = sorted(set(groups))
    rng = np.random.default_rng(SEED)
    day_draws = [rng.choice(unique_days, size=len(unique_days), replace=True) for _ in range(N_BOOTSTRAP)]

    base_aucs_gbt = replicate_aucs_for_draws(X_base, y, groups, _gbt_fit_predict, day_draws)
    therm_aucs_gbt = replicate_aucs_for_draws(X_therm, y, groups, _gbt_fit_predict, day_draws)
    both_valid = ~np.isnan(base_aucs_gbt) & ~np.isnan(therm_aucs_gbt)
    diffs_gbt = therm_aucs_gbt[both_valid] - base_aucs_gbt[both_valid]
    print(f"  GBT: n_paired={both_valid.sum()}/{N_BOOTSTRAP}  mean_diff={diffs_gbt.mean():+.4f}"
          f"  95% CI=[{np.percentile(diffs_gbt, 2.5):+.4f}, {np.percentile(diffs_gbt, 97.5):+.4f}]"
          f"  % resamples favoring thermal={100 * (diffs_gbt > 0).mean():.1f}%")

    base_aucs_lr = replicate_aucs_for_draws(X_base, y, groups, _lr_fit_predict, day_draws)
    therm_aucs_lr = replicate_aucs_for_draws(X_therm, y, groups, _lr_fit_predict, day_draws)
    both_valid_lr = ~np.isnan(base_aucs_lr) & ~np.isnan(therm_aucs_lr)
    diffs_lr = therm_aucs_lr[both_valid_lr] - base_aucs_lr[both_valid_lr]
    print(f"  LR:  n_paired={both_valid_lr.sum()}/{N_BOOTSTRAP}  mean_diff={diffs_lr.mean():+.4f}"
          f"  95% CI=[{np.percentile(diffs_lr, 2.5):+.4f}, {np.percentile(diffs_lr, 97.5):+.4f}]"
          f"  % resamples favoring thermal={100 * (diffs_lr > 0).mean():.1f}%")

    results = {
        "n_windows": int(X_base.shape[0]),
        "n_features_base": int(X_base.shape[1]),
        "n_features_thermal": int(X_therm.shape[1]),
        "new_thermal_features": new_thermal_features,
        "days": unique_days,
        "cv_gbt": {"no_thermal": cv_base_gbt, "with_thermal": cv_therm_gbt},
        "cv_lr":  {"no_thermal": cv_base_lr,  "with_thermal": cv_therm_lr},
        "paired_bootstrap_gbt": {
            "n_bootstrap": N_BOOTSTRAP, "n_paired": int(both_valid.sum()),
            "mean_diff": float(diffs_gbt.mean()),
            "ci_low": float(np.percentile(diffs_gbt, 2.5)),
            "ci_high": float(np.percentile(diffs_gbt, 97.5)),
            "pct_resamples_favoring_thermal": float(100 * (diffs_gbt > 0).mean()),
        },
        "paired_bootstrap_lr": {
            "n_bootstrap": N_BOOTSTRAP, "n_paired": int(both_valid_lr.sum()),
            "mean_diff": float(diffs_lr.mean()),
            "ci_low": float(np.percentile(diffs_lr, 2.5)),
            "ci_high": float(np.percentile(diffs_lr, 97.5)),
            "pct_resamples_favoring_thermal": float(100 * (diffs_lr > 0).mean()),
        },
    }
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
