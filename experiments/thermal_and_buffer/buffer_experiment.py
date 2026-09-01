"""
buffer_experiment.py - does the pre-event buffer (currently 7000ms, see
host/training/build_dataset.py's DEFAULT_BUFFER_MS) matter? Tests 7s
(current), 5s, and 3s.

Pre-registered plan (written before looking at results):
  1. For each buffer_ms in [7000, 5000, 3000], build a full dataset via
     host.training.build_dataset.build_dataset(buffer_ms=...) - reused
     unmodified, standard 60s window, standard compute_features(). Only
     buffer_ms changes; everything else is the current pipeline as-is.
  2. Score each with day-based LOGO CV using GradientBoostingClassifier
     (n_estimators=200, max_depth=3, random_state=42) - the actual best
     model in host/training/train.py today.
  3. Paired day-block bootstrap, 5s vs 7s and 3s vs 7s, using the SAME
     sequence of day-draws for both sides of each comparison.

Caveat, stated up front: unlike thermal_experiment.py, this is
NOT a matched-population comparison. Changing buffer_ms shifts which
60s window is anchored to each marker, which changes the actual sample
rows (not just which features are computed from them) and can change how
many windows survive the coverage/exclusion-zone filtering. All three
buffer_ms settings span the same calendar days, so "pair by which days
got drawn" is still well-defined (cancels shared day-to-day noise), but
it is NOT pairing on identical rows - same caveat class as
experiments/model_sweep/paired_bootstrap.py's B/D-vs-A comparisons.

host/pipeline/, host/training/ are imported read-only. Nothing there is
modified.

Usage:
    python -m experiments.thermal_and_buffer.buffer_experiment
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from host.training.build_dataset import build_dataset
from host.training.cross_validation import day_based_cv

OUT_PATH = Path(__file__).parent / "buffer_results.json"
N_BOOTSTRAP = 100  # reduced from 200 - the full run (3 dataset builds x 2
                    # candidates x GBT refits) took too long at 200
SEED = 42
BUFFER_VALUES_MS = [7000, 5000, 3000]  # 7000 = current default, first = reference


def _gbt_fit_predict(X_tr, y_tr, X_te):
    scaler = StandardScaler()
    X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
    clf.fit(X_tr_s, y_tr)
    return clf.predict_proba(X_te_s)[:, 1]


def replicate_aucs_for_draws(X, y, groups, fit_predict_fn, day_draws) -> np.ndarray:
    """Duplicated from thermal_experiment.py rather than imported, to keep
    this experiment self-contained."""
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
    datasets = {}
    for buffer_ms in BUFFER_VALUES_MS:
        print(f"\nBuilding dataset for buffer_ms={buffer_ms} ...")
        X, y, groups, names = build_dataset(buffer_ms=buffer_ms)
        datasets[buffer_ms] = (X, y, groups, names)

    ref_days = sorted(set(datasets[BUFFER_VALUES_MS[0]][2]))
    for b in BUFFER_VALUES_MS[1:]:
        days_b = sorted(set(datasets[b][2]))
        if days_b != ref_days:
            print(f"  NOTE: buffer_ms={b} spans different days ({days_b}) than "
                  f"buffer_ms={BUFFER_VALUES_MS[0]} ({ref_days}) - day-draw pairing "
                  f"restricted to the intersection.")

    print("\n--- Day-based LOGO CV (GBT, current best model) per buffer_ms ---")
    cv_results = {}
    for buffer_ms in BUFFER_VALUES_MS:
        X, y, groups, names = datasets[buffer_ms]
        result = day_based_cv(X, y, groups, GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42))
        cv_results[buffer_ms] = result
        print(f"  buffer_ms={buffer_ms:>5}  n={X.shape[0]:>4}  positive_rate={y.mean():.2%}"
              f"  AUC={result['mean_auc']:.3f} +/- {result['std_auc']:.3f}")

    print(f"\n--- Paired day-block bootstrap vs. buffer_ms={BUFFER_VALUES_MS[0]} (N={N_BOOTSTRAP}) ---")
    print("  CROSS-POPULATION pairing (see caveat in module docstring) - pairs on")
    print("  shared day-draws, not identical rows.")

    ref_buffer = BUFFER_VALUES_MS[0]
    X_ref, y_ref, groups_ref, _ = datasets[ref_buffer]
    unique_days_ref = sorted(set(groups_ref))
    rng = np.random.default_rng(SEED)
    day_draws = [rng.choice(unique_days_ref, size=len(unique_days_ref), replace=True) for _ in range(N_BOOTSTRAP)]

    ref_aucs = replicate_aucs_for_draws(X_ref, y_ref, groups_ref, _gbt_fit_predict, day_draws)

    paired_results = {}
    for buffer_ms in BUFFER_VALUES_MS[1:]:
        X, y, groups, _ = datasets[buffer_ms]
        # Restrict this buffer's day-draws to days it actually has, per-draw,
        # so a draw containing a day absent from this buffer's dataset
        # doesn't spuriously fail the whole replicate.
        cand_aucs = replicate_aucs_for_draws(X, y, groups, _gbt_fit_predict, day_draws)
        both_valid = ~np.isnan(ref_aucs) & ~np.isnan(cand_aucs)
        diffs = cand_aucs[both_valid] - ref_aucs[both_valid]
        pct_beat = float(100 * (diffs > 0).mean()) if len(diffs) else float("nan")
        print(f"  buffer_ms={buffer_ms} vs {ref_buffer}: n_paired={both_valid.sum()}/{N_BOOTSTRAP}"
              f"  mean_diff={diffs.mean():+.4f}"
              f"  95% CI=[{np.percentile(diffs, 2.5):+.4f}, {np.percentile(diffs, 97.5):+.4f}]"
              f"  % resamples favoring buffer_ms={buffer_ms}: {pct_beat:.1f}%")
        paired_results[str(buffer_ms)] = {
            "n_paired": int(both_valid.sum()),
            "mean_diff": float(diffs.mean()),
            "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)),
            "pct_resamples_favoring_candidate": pct_beat,
        }

    results = {
        "reference_buffer_ms": ref_buffer,
        "n_bootstrap": N_BOOTSTRAP,
        "cv_results": {str(k): v for k, v in cv_results.items()},
        "n_samples": {str(k): int(v[0].shape[0]) for k, v in datasets.items()},
        "positive_rate": {str(k): float(v[1].mean()) for k, v in datasets.items()},
        "days": {str(k): sorted(set(v[2])) for k, v in datasets.items()},
        "paired_bootstrap_vs_reference": paired_results,
    }
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
