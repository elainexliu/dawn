"""
run_experiment.py - day-based CV comparison of IMU-only vs. IMU + clean-
early-segment HR trend / HRV, plus the quality-funnel report.

day_based_cv is reused unmodified from host/training/cross_validation.py.
The day-block bootstrap (only run if the common-subset sample size ends up
close enough to the full baseline's - see main()) is implemented fresh
here rather than imported from experiments/informed_prior/evaluate.py,
since this experiment is meant to be isolated from the other experiment
folders, not just from experiments/model_sweep/'s comparisons specifically.

Usage:
    python -m experiments.hrv_hr_early_window.run_experiment
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from experiments.hrv_hr_early_window.build_datasets import build_all_configs
from host.training.cross_validation import day_based_cv

OUT_DIR = Path(__file__).parent
SIZE_MISMATCH_TOLERANCE = 0.30  # bootstrap only if common subset is within 30% of the full baseline's n
N_BOOTSTRAP = 200


def day_block_bootstrap(X, y, groups, fit_predict_fn, n_bootstrap=N_BOOTSTRAP, seed=42) -> dict:
    """Resample the set of unique days with replacement, refit per
    resample - same algorithm used elsewhere in this project
    (day_block_bootstrap in experiments/informed_prior/evaluate.py),
    reimplemented locally so this experiment has no cross-experiment
    dependency."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    unique_days = np.array(sorted(set(groups)))
    n_days = len(unique_days)
    replicate_means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(unique_days, size=n_days, replace=True)
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
            replicate_means.append(float(np.mean(fold_aucs)))

    replicate_means = np.array(replicate_means)
    if len(replicate_means) == 0:
        return {"n_valid_replicates": 0, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    return {
        "n_valid_replicates": int(len(replicate_means)),
        "mean": float(replicate_means.mean()),
        "ci_low": float(np.percentile(replicate_means, 2.5)),
        "ci_high": float(np.percentile(replicate_means, 97.5)),
    }


def _lr_fit_predict(X_tr, y_tr, X_te):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)
    clf = LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced").fit(X_tr_s, y_tr)
    return clf.predict_proba(X_te_s)[:, 1]


def main():
    (X_a_full, y_a_full, groups_a_full, names_a_full), restricted, quality_report, quality_log = build_all_configs()

    print("\n=== Reference: full-size feature-set-A baseline (60s window, all valid anchors) ===")
    ref_cv = day_based_cv(X_a_full, y_a_full, groups_a_full, LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced"))
    print(f"  n={X_a_full.shape[0]}  AUC={ref_cv['mean_auc']:.3f}+/-{ref_cv['std_auc']:.3f}  F1={ref_cv['mean_f1']:.3f}")

    n_common = quality_report["n_both_ok"]
    n_full = quality_report["n_full_reference_A"]
    size_ratio = n_common / n_full if n_full else 0
    print(f"\nCommon-subset sample size: {n_common} vs. full baseline's {n_full} "
          f"({size_ratio:.1%} of baseline)")

    print("\n=== Four configs on the common subset (fair, apples-to-apples) ===")
    results = {"reference_full_A": {"n": int(X_a_full.shape[0]), **ref_cv}, "quality_report": quality_report,
               "configs": {}}
    for name, (X, y, groups) in restricted.items():
        cv = day_based_cv(X, y, groups, LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced"))
        print(f"  {name:<18} n={X.shape[0]:<4} n_features={X.shape[1]:<3} "
              f"AUC={cv['mean_auc']:.3f}+/-{cv['std_auc']:.3f}  F1={cv['mean_f1']:.3f}")
        results["configs"][name] = {"n": int(X.shape[0]), "n_features": int(X.shape[1]), **cv}

    should_bootstrap = size_ratio >= (1 - SIZE_MISMATCH_TOLERANCE)
    print(f"\nCommon subset is {'within' if should_bootstrap else 'NOT within'} "
          f"{SIZE_MISMATCH_TOLERANCE:.0%} of the full baseline's size - "
          f"{'proceeding with' if should_bootstrap else 'SKIPPING'} bootstrap.")

    if should_bootstrap:
        print("\n=== Day-block bootstrap (common subset only, since sizes are close enough) ===")
        boot_results = {}
        for name, (X, y, groups) in restricted.items():
            boot = day_block_bootstrap(X, y, groups, _lr_fit_predict, n_bootstrap=N_BOOTSTRAP)
            boot_results[name] = boot
            print(f"  {name:<18} mean={boot['mean']:.3f}  95% CI=[{boot['ci_low']:.3f}, {boot['ci_high']:.3f}]")
        results["bootstrap"] = boot_results
    else:
        results["bootstrap"] = None
        print("  (Bootstrapping a sample this much smaller than the reference baseline would reintroduce")
        print("   the same population-mismatch problem flagged for feature sets B/D in model_sweep -")
        print("   skipped rather than reported as if it were a clean comparison.)")

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved: {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
