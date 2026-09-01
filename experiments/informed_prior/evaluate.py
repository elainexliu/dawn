"""
evaluate.py - compares the zero-shrinkage baseline against the
informed-prior model, apples-to-apples, on personal data.

The population prior only covers a subset of the 54 accel/gyro features
(see population_prior.json's excluded_features - Nyquist limits and
cross-dataset scale issues in the FFT-derived stats), so there are two
baseline numbers here, not one:

  (a) sanity-check reproduction: current pipeline as-is, full 54 features.
      Checked against the previously-reported 0.739+/-0.059 AUC, which is
      stale (measured on a 3-day/132-window dataset that's since grown to
      5 days/217 windows) - won't reproduce exactly, as expected.
  (b) fair comparison baseline: the same model restricted to the features
      the prior actually covers - this is what the informed-prior sweep
      below is measured against.

Prior unit conversion: population_prior.json's coefficients are in
population standard-deviation units, but each personal CV fold
standardizes on its own mean/std, so comparing them needs a conversion
through raw physical units:
    beta_raw = beta_population_std / population_scaler.scale_
    beta_this_fold_units = beta_raw * this_fold_scaler.scale_
informed_day_based_cv() mirrors day_based_cv()'s splitting/scaling/metric
logic exactly, adding only this per-fold prior rescaling (day_based_cv
itself has no hook for a prior that varies per fold).

Usage:
    python -m experiments.informed_prior.evaluate
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from experiments.informed_prior.informed_prior_model import InformedPriorLogisticRegression
from host.training.build_dataset import DEFAULT_X_WINDOW_MS, _min_samples_for_window, collect_windows, windows_to_features
from host.training.cross_validation import day_based_cv

PRIOR_PATH = Path(__file__).parent / "population_prior.json"
OUT_JSON   = Path(__file__).parent / "evaluation_results.json"

STALE_REFERENCE_AUC = 0.739
STALE_REFERENCE_STD = 0.059

LAM_SWEEP = [1, 5, 20, 50, 100, 300, 1000, 5000]
# ~50 is roughly where sklearn's C=0.01 sits in this parameterization
# (sklearn's L2 penalty scales roughly as 1/(2C)) - i.e. "similar total
# regularization budget to the current baseline, just aimed at the
# population prior instead of zero." The sweep spans well below and above
# that reference point.
N_BOOTSTRAP = 1000
SEED = 42


def _select_accel_gyro(feature_names: list[str]) -> list[int]:
    return [i for i, n in enumerate(feature_names) if n.startswith(("accel_", "gyro_"))]


def _load_personal_data():
    windows = collect_windows()  # defaults: n_lookback_shifts=1 = current pipeline's exact windowing
    min_samples = _min_samples_for_window(DEFAULT_X_WINDOW_MS)
    X, y, groups, feature_names = windows_to_features(windows, min_samples)
    idx = _select_accel_gyro(feature_names)
    return X[:, idx], y, groups, [feature_names[i] for i in idx]


def _restrict_to_population_features(X, feature_names, population_feature_names):
    """Align personal data's columns to the population's 48-feature set,
    by name (never by position) - verifies the schemas actually match
    rather than assuming it.
    """
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    missing = [n for n in population_feature_names if n not in name_to_idx]
    if missing:
        raise ValueError(f"Personal feature set is missing features the population model expects: {missing}")
    idx = [name_to_idx[n] for n in population_feature_names]
    return X[:, idx]


def informed_day_based_cv(X, y, groups, pop_coef_raw, lam) -> dict:
    """Mirrors host.training.cross_validation.day_based_cv's exact LOGO /
    per-fold-scaler / metrics logic; the one addition is re-expressing the
    prior in each fold's own standardized units before fitting - see
    module docstring for why a static prior can't just be passed into the
    existing day_based_cv unchanged."""
    logo = LeaveOneGroupOut()
    aucs, f1s = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        if len(np.unique(y_te)) < 2:
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        fold_prior = pop_coef_raw * scaler.scale_

        clf = InformedPriorLogisticRegression(prior=fold_prior, lam=lam).fit(X_tr_s, y_tr)
        y_prob = clf.predict_proba(X_te_s)[:, 1]
        y_pred = clf.predict(X_te_s)
        aucs.append(roc_auc_score(y_te, y_prob))
        f1s.append(f1_score(y_te, y_pred, zero_division=0))

    aucs_arr, f1s_arr = np.array(aucs), np.array(f1s)
    return {
        "mean_auc": float(aucs_arr.mean()) if len(aucs_arr) else float("nan"),
        "std_auc":  float(aucs_arr.std())  if len(aucs_arr) else float("nan"),
        "mean_f1":  float(f1s_arr.mean())  if len(f1s_arr) else float("nan"),
        "std_f1":   float(f1s_arr.std())   if len(f1s_arr) else float("nan"),
        "fold_aucs": aucs_arr.tolist(),
        "fold_f1s":  f1s_arr.tolist(),
    }


def day_block_bootstrap(X, y, groups, fit_predict_fn, n_bootstrap=N_BOOTSTRAP, seed=SEED) -> dict:
    """Resample the SET of unique days with replacement (n draws, n = number
    of unique days), refit per resample. For each unique day drawn, hold
    it out as test EXACTLY ONCE and train on the pooled data from every
    OTHER day in the resample (each repeated as many times as it was
    drawn, reflecting the bootstrap weighting) - a day drawn multiple
    times is still only ever tested once per replicate, and is fully
    excluded from that replicate's training pool to avoid the same day's
    data appearing in both train and test.

    fit_predict_fn(X_tr, y_tr, X_te) -> y_prob_te : a closure that does its
    own scaling/fitting appropriate to whichever model variant is being
    bootstrapped (baseline or informed-prior).

    Returns replicate-level mean AUCs and their 2.5/97.5 percentile CI.
    """
    rng = np.random.default_rng(seed)
    unique_days = np.array(sorted(set(groups)))
    n_days = len(unique_days)
    replicate_means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(unique_days, size=n_days, replace=True)
        unique_in_sample = sorted(set(sample))
        if len(unique_in_sample) < 2:
            continue  # no "other" days to train on

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


def main():
    prior = json.loads(PRIOR_PATH.read_text())
    pop_feature_names = prior["feature_names"]
    pop_coef_std = np.array(prior["coefficients_population_std_units"])
    pop_scale = np.array(prior["population_scaler_scale"])
    pop_coef_raw = pop_coef_std / pop_scale  # -> raw physical-unit coefficients

    print("Loading personal data (host.training.build_dataset, unmodified) ...")
    X_full, y, groups, feature_names_full = _load_personal_data()
    print(f"  X={X_full.shape}  positive_rate={y.mean():.2%}  days={sorted(set(groups))}")

    # --- (a) Sanity-check reproduction: full 54 features, current model as-is ---
    print("\n--- (a) Sanity-check reproduction: current model, full 54 features ---")
    baseline_full_cv = day_based_cv(X_full, y, groups, LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced"))
    print(f"  AUC={baseline_full_cv['mean_auc']:.3f}+/-{baseline_full_cv['std_auc']:.3f}"
          f"  (stale reference from an earlier, smaller dataset: {STALE_REFERENCE_AUC}+/-{STALE_REFERENCE_STD})")
    print("  Mismatch vs. the stale reference is expected - the dataset has grown from 3 days/132 windows"
          " to 5 days/217 windows since that number was recorded; this is the CURRENT true baseline, not a bug.")

    # --- Verify schema alignment, restrict personal data to the 48-feature population-covered set ---
    print(f"\nPopulation prior covers {len(pop_feature_names)} features"
          f" (excluded: {prior['excluded_features']})")
    X_48 = _restrict_to_population_features(X_full, feature_names_full, pop_feature_names)
    print(f"  Verified: all {len(pop_feature_names)} population feature names present in personal data. "
          f"X restricted to {X_48.shape}.")

    # --- (b) Fair comparison baseline: SAME model, same feature subset as the prior ---
    print(f"\n--- (b) Fair comparison baseline: LogisticRegression(C=0.01), {len(pop_feature_names)} features ---")
    baseline_48_cv = day_based_cv(X_48, y, groups, LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced"))
    print(f"  AUC={baseline_48_cv['mean_auc']:.3f}+/-{baseline_48_cv['std_auc']:.3f}"
          f"  F1={baseline_48_cv['mean_f1']:.3f}+/-{baseline_48_cv['std_f1']:.3f}")

    # --- (c) Informed-prior sweep ---
    print(f"\n--- (c) Informed-prior sweep, lam in {LAM_SWEEP} ---")
    sweep_results = {}
    for lam in LAM_SWEEP:
        cv = informed_day_based_cv(X_48, y, groups, pop_coef_raw, lam)
        sweep_results[lam] = cv
        print(f"  lam={lam:<6}  AUC={cv['mean_auc']:.3f}+/-{cv['std_auc']:.3f}  F1={cv['mean_f1']:.3f}+/-{cv['std_f1']:.3f}")

    best_lam = max(sweep_results, key=lambda l: sweep_results[l]["mean_auc"])
    print(f"\nBest-performing lam by mean CV AUC: {best_lam}"
          f"  (AUC={sweep_results[best_lam]['mean_auc']:.3f}+/-{sweep_results[best_lam]['std_auc']:.3f})")

    # --- Day-level block bootstrap: baseline (b) vs. best informed-prior (c) ---
    print(f"\n--- Day-level block bootstrap ({N_BOOTSTRAP} resamples) ---")

    def _baseline_fit_predict(X_tr, y_tr, X_te):
        scaler = StandardScaler()
        X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)
        clf = LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced").fit(X_tr_s, y_tr)
        return clf.predict_proba(X_te_s)[:, 1]

    def _informed_fit_predict(X_tr, y_tr, X_te):
        scaler = StandardScaler()
        X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)
        fold_prior = pop_coef_raw * scaler.scale_
        clf = InformedPriorLogisticRegression(prior=fold_prior, lam=best_lam).fit(X_tr_s, y_tr)
        return clf.predict_proba(X_te_s)[:, 1]

    boot_baseline = day_block_bootstrap(X_48, y, groups, _baseline_fit_predict)
    boot_informed = day_block_bootstrap(X_48, y, groups, _informed_fit_predict)
    print(f"  baseline (48 feat):      mean={boot_baseline['mean']:.3f}  "
          f"95% CI=[{boot_baseline['ci_low']:.3f}, {boot_baseline['ci_high']:.3f}]"
          f"  (n_valid_replicates={boot_baseline['n_valid_replicates']}/{N_BOOTSTRAP})")
    print(f"  informed prior (lam={best_lam}): mean={boot_informed['mean']:.3f}  "
          f"95% CI=[{boot_informed['ci_low']:.3f}, {boot_informed['ci_high']:.3f}]"
          f"  (n_valid_replicates={boot_informed['n_valid_replicates']}/{N_BOOTSTRAP})")

    overlap = not (boot_informed["ci_high"] < boot_baseline["ci_low"] or boot_baseline["ci_high"] < boot_informed["ci_low"])
    print(f"  CIs {'OVERLAP - no statistically distinguishable winner at this sample size' if overlap else 'DO NOT overlap'}")

    # --- Coefficient comparison at best lam, fit once on all personal data ---
    print(f"\n--- Coefficient shift at lam={best_lam}, fit on all personal data ---")
    scaler_full = StandardScaler()
    X_48_scaled = scaler_full.fit_transform(X_48)
    baseline_full_fit = LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced").fit(X_48_scaled, y)
    fold_prior_full = pop_coef_raw * scaler_full.scale_
    informed_full_fit = InformedPriorLogisticRegression(prior=fold_prior_full, lam=best_lam).fit(X_48_scaled, y)

    coef_diffs = []
    for i, name in enumerate(pop_feature_names):
        b_base = baseline_full_fit.coef_[0][i]
        b_inf = informed_full_fit.coef_[0][i]
        coef_diffs.append({"feature": name, "baseline_coef": float(b_base), "informed_coef": float(b_inf),
                           "prior_coef_fold_units": float(fold_prior_full[i]), "abs_diff": float(abs(b_inf - b_base))})
    coef_diffs.sort(key=lambda d: -d["abs_diff"])
    for d in coef_diffs[:10]:
        print(f"  {d['feature']:<28} baseline={d['baseline_coef']:+.3f}  informed={d['informed_coef']:+.3f}"
              f"  prior={d['prior_coef_fold_units']:+.3f}  |diff|={d['abs_diff']:.3f}")

    results = {
        "sanity_check_full54": baseline_full_cv,
        "stale_reference": {"auc": STALE_REFERENCE_AUC, "std": STALE_REFERENCE_STD},
        "fair_baseline_48": baseline_48_cv,
        "lam_sweep": {str(k): v for k, v in sweep_results.items()},
        "best_lam": best_lam,
        "bootstrap_baseline": boot_baseline,
        "bootstrap_informed": boot_informed,
        "cis_overlap": overlap,
        "coefficient_diffs": coef_diffs,
    }
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
