"""
paired_bootstrap.py - paired day-level block bootstrap: A_only vs.
A_plus_hrv, same methodology as model_sweep/paired_bootstrap.py: marginal
CIs conflate shared "which days are hard" noise with the actual
candidate-vs-baseline skill difference, so pairing on the same day-draw
sequence cancels the shared part and isolates the second.

Unlike model_sweep's B/D candidates, A_only and A_plus_hrv aren't a
cross-population comparison - the verification block in main() checks this
programmatically (identical y, identical groups, A_plus_hrv's first 54
columns exactly equal to A_only's full matrix) rather than assuming it.
Raises if any check fails rather than proceeding with an invalid pairing.

Usage:
    python -m experiments.hrv_hr_early_window.paired_bootstrap
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from experiments.hrv_hr_early_window.build_datasets import build_all_configs

N_BOOTSTRAP = 500  # both configs use cheap LogisticRegression(C=0.01) only -
                    # no RandomForest bottleneck like model_sweep hit, so the
                    # full 500 is tractable directly.
SEED = 42
OUT_PATH = Path(__file__).parent / "paired_bootstrap_results.json"


def build_shared_day_draws(unique_days: list[str], n_bootstrap: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.choice(unique_days, size=len(unique_days), replace=True) for _ in range(n_bootstrap)]


def replicate_aucs_for_draws(X, y, groups, fit_predict_fn, day_draws: list[np.ndarray]) -> np.ndarray:
    """Identical algorithm to experiments/model_sweep/paired_bootstrap.py's
    function of the same name (itself mirroring day_block_bootstrap's
    internals) - resample unique days with replacement, test each unique
    day in the draw once, train on the rest weighted by draw count."""
    replicate_means = np.full(len(day_draws), np.nan)
    for i, sample in enumerate(day_draws):
        unique_in_sample = sorted(set(sample))
        if len(unique_in_sample) < 2:
            continue
        fold_aucs = []
        for test_day in unique_in_sample:
            train_days_multiset = [d for d in sample if d != test_day]
            if not train_days_multiset:
                continue
            train_idx_pool = []
            for d in train_days_multiset:
                train_idx_pool.extend(np.flatnonzero(groups == d))
            test_idx = np.flatnonzero(groups == test_day)
            y_tr, y_te = y[train_idx_pool], y[test_idx]
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue
            y_prob = fit_predict_fn(X[train_idx_pool], y_tr, X[test_idx])
            fold_aucs.append(roc_auc_score(y_te, y_prob))
        if fold_aucs:
            replicate_means[i] = float(np.mean(fold_aucs))
    return replicate_means


def _lr_fit_predict(X_tr, y_tr, X_te):
    scaler = StandardScaler()
    X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)
    clf = LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced").fit(X_tr_s, y_tr)
    return clf.predict_proba(X_te_s)[:, 1]


def main():
    print("Rebuilding datasets via build_datasets.build_all_configs() (imported, not modified) ...")
    (X_a_full, y_a_full, groups_a_full, names_a_full), restricted, quality_report, quality_log = build_all_configs()

    X_only, y_only, groups_only = restricted["A_only"]
    X_hrv, y_hrv, groups_hrv = restricted["A_plus_hrv"]

    print(f"\nDataset drift check: common subset is now {X_only.shape[0]} samples "
          f"(results.md reported 193 at the time it was written - expected to drift as new "
          f"sessions land in data/raw/, same as every other dataset in this project).")

    # --- Verify the two configs share the same rows rather than assume it ---
    print("\nVerifying A_only and A_plus_hrv share the identical row population ...")
    same_n = X_only.shape[0] == X_hrv.shape[0]
    same_y = same_n and np.array_equal(y_only, y_hrv)
    same_groups = same_n and np.array_equal(groups_only, groups_hrv)
    same_imu_block = same_n and np.allclose(X_hrv[:, :X_only.shape[1]], X_only)
    hrv_is_superset = X_hrv.shape[1] == X_only.shape[1] + 2  # +RMSSD, +SDNN

    print(f"  same n:               {same_n}  ({X_only.shape[0]} vs {X_hrv.shape[0]})")
    print(f"  same y (elementwise):  {same_y}")
    print(f"  same groups:           {same_groups}")
    print(f"  A_plus_hrv's IMU block matches A_only exactly: {same_imu_block}")
    print(f"  A_plus_hrv = A_only + exactly 2 extra columns: {hrv_is_superset}")

    if not (same_n and same_y and same_groups and same_imu_block and hrv_is_superset):
        raise RuntimeError(
            "A_only and A_plus_hrv do NOT share the identical row population - "
            "stopping rather than proceeding with an invalid paired comparison. "
            f"same_n={same_n} same_y={same_y} same_groups={same_groups} "
            f"same_imu_block={same_imu_block} hrv_is_superset={hrv_is_superset}"
        )
    print("  CONFIRMED: identical population - this is a clean, fully matched paired comparison,")
    print("  same tier as informed_prior_flipped vs. baseline in model_sweep (not a B/D-style cross-population one).")

    unique_days = sorted(set(groups_only))
    print(f"\nGenerating {N_BOOTSTRAP} shared day-draws (seed={SEED}), days={unique_days} ...")
    day_draws = build_shared_day_draws(unique_days, N_BOOTSTRAP, SEED)

    print("\nComputing A_only per-resample AUCs ...")
    a_only_aucs = replicate_aucs_for_draws(X_only, y_only, groups_only, _lr_fit_predict, day_draws)
    print(f"  {int(np.sum(~np.isnan(a_only_aucs)))}/{N_BOOTSTRAP} valid")

    print("\nComputing A_plus_hrv per-resample AUCs (same day_draws) ...")
    a_hrv_aucs = replicate_aucs_for_draws(X_hrv, y_hrv, groups_hrv, _lr_fit_predict, day_draws)
    print(f"  {int(np.sum(~np.isnan(a_hrv_aucs)))}/{N_BOOTSTRAP} valid")

    both_valid = ~np.isnan(a_only_aucs) & ~np.isnan(a_hrv_aucs)
    n_paired = int(both_valid.sum())
    n_dropped = N_BOOTSTRAP - n_paired
    diffs = a_hrv_aucs[both_valid] - a_only_aucs[both_valid]

    mean_diff = float(diffs.mean())
    ci_low, ci_high = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    pct_beat = float((diffs > 0).mean() * 100)

    print(f"\n--- Paired diff (A_plus_hrv - A_only) ---")
    print(f"  n_paired={n_paired}/{N_BOOTSTRAP}  (dropped: {n_dropped})")
    print(f"  mean_diff={mean_diff:+.4f}")
    print(f"  95% CI=[{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"  % resamples A_plus_hrv beat A_only: {pct_beat:.1f}%")

    significant = ci_low > 0
    print(f"\n  {'CI EXCLUDES ZERO - statistically significant at 95%' if significant else 'CI still includes zero - not significant at 95%'}")

    results = {
        "day_draws_seed": SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "unique_days": unique_days,
        "n_samples_common_subset": int(X_only.shape[0]),
        "verification": {
            "same_n": same_n, "same_y": same_y, "same_groups": same_groups,
            "same_imu_block": same_imu_block, "hrv_is_superset": hrv_is_superset,
        },
        "n_paired": n_paired,
        "n_dropped": n_dropped,
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "pct_resamples_beat": pct_beat,
        "significant_95": significant,
        "a_only_aucs": a_only_aucs.tolist(),
        "a_plus_hrv_aucs": a_hrv_aucs.tolist(),
    }
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
