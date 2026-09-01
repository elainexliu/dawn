"""
paired_bootstrap.py - paired day-level block bootstrap: baseline vs. each
top candidate, using the SAME sequence of 500 day-draws for every config
so the comparison is per-resample apples-to-apples, not just
marginal-CI-overlap (which is what results.md originally reported - marginal
CIs can overlap even when a config wins on most individual resamples, if
the shared "easy/hard day" noise dominates both).

bootstrap_results.json and bootstrap_informed_prior_flipped.json only
store aggregated stats (mean, ci_low, ci_high) - day_block_bootstrap never
returns the per-replicate array, so there's nothing to pair post-hoc. This
script reruns fresh, generating one shared list of day-draws up front and
reusing it for every config.

Caveat on B/D vs. baseline pairing, worth reading before trusting those
numbers: baseline (feature set A) has 245 samples; B/D have only 174
(different sample population - the 180s window needed for B/D's longest
horizon can't be satisfied near session starts, and that survivorship
isn't symmetric between classes, see results.md). A/B/D all span the
identical 5 calendar days, so "pair by which days got drawn" is still
well-defined for all three - but this isn't a matched-pairs design in the
strict sense (same rows scored under both conditions); it's "same days
sampled," not "same windows sampled." It cancels shared between-day noise
(e.g. day 3 was unusually easy/hard for everyone) but not noise from the
different within-day sample composition each population sees. Flagged
per-candidate below.

The ONE clean, fully matched comparison here is informed_prior_flipped vs.
baseline: both operate on feature set A (informed_prior_flipped restricted
to A's 36-feature subset) - identical 245-row population, so that pairing
has no such caveat.

Usage:
    python -m experiments.model_sweep.paired_bootstrap
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from experiments.informed_prior.informed_prior_model import InformedPriorLogisticRegression
from experiments.model_sweep.feature_variants import FEATURE_SET_BUILDERS

# Reduced from 500 to 200 after repeated kills - RandomForest refit 500x
# per candidate exceeded the available runtime even from cached data.
# Still large enough for a meaningful percentile CI; noted explicitly in
# the report as a compute-driven reduction, not a methodology change.
N_BOOTSTRAP = 200
SEED = 42
OUT_PATH = Path(__file__).parent / "paired_bootstrap_results.json"


def build_shared_day_draws(unique_days: list[str], n_bootstrap: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.choice(unique_days, size=len(unique_days), replace=True) for _ in range(n_bootstrap)]


def replicate_aucs_for_draws(X, y, groups, fit_predict_fn, day_draws: list[np.ndarray]) -> np.ndarray:
    """Same algorithm as experiments/informed_prior/evaluate.py's
    day_block_bootstrap, but takes a pre-generated draw sequence (so it can
    be shared across configs) and returns the per-replicate array instead
    of only the aggregate - that return-value difference is exactly why
    this can't just call the existing function unmodified.
    """
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


def _nb_fit_predict(X_tr, y_tr, X_te):
    scaler = StandardScaler()
    X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)
    clf = GaussianNB().fit(X_tr_s, y_tr)
    return clf.predict_proba(X_te_s)[:, 1]


def _rf_fit_predict(X_tr, y_tr, X_te):
    scaler = StandardScaler()
    X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)
    clf = RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=8,
                                 class_weight="balanced", random_state=42).fit(X_tr_s, y_tr)
    return clf.predict_proba(X_te_s)[:, 1]


def _cache_path(fs_name: str) -> Path:
    return Path(__file__).parent / f"_dataset_cache_{fs_name}.npz"


def _build_one_cached(fs_name: str):
    """Per-feature-set cache, built and saved independently - raw-packet
    parsing (FEATURE_SET_BUILDERS) re-scans every session from scratch and
    has been the actual bottleneck behind repeated kills here (compounded
    by new sessions landing in data/raw/ mid-run, and by data volume
    growing enough that building all of A+B+D in one process now exceeds
    the available runtime). Splitting per-feature-set means a kill costs
    at most one feature set's build, not all three. Delete a cache file
    manually to pick up new raw data for that feature set.
    """
    path = _cache_path(fs_name)
    if path.exists():
        print(f"  {fs_name}: loading cache from {path} ...")
        d = np.load(path, allow_pickle=True)
        return d["X"], d["y"], d["groups"], list(d["names"])

    print(f"  {fs_name}: no cache found, building ...")
    X, y, groups, names, _ = FEATURE_SET_BUILDERS[fs_name]()
    np.savez(path, X=X, y=y, groups=groups, names=np.array(names))
    print(f"  {fs_name}: cached to {path}")
    return X, y, groups, names


def _build_datasets_cached():
    X_a, y_a, groups_a, names_a = _build_one_cached("A")
    X_b, y_b, groups_b, names_b = _build_one_cached("B")
    X_d, y_d, groups_d, names_d = _build_one_cached("D")
    return X_a, y_a, groups_a, names_a, X_b, y_b, groups_b, X_d, y_d, groups_d


def main():
    X_a, y_a, groups_a, names_a, X_b, y_b, groups_b, X_d, y_d, groups_d = _build_datasets_cached()
    print(f"  A: {X_a.shape}  days={sorted(set(groups_a))}")
    print(f"  B: {X_b.shape}  days={sorted(set(groups_b))}")
    print(f"  D: {X_d.shape}  days={sorted(set(groups_d))}")
    assert sorted(set(groups_a)) == sorted(set(groups_b)) == sorted(set(groups_d)), \
        "feature sets span different calendar days - day-level pairing would be invalid"

    prior_flipped = json.loads(Path("experiments/diagnostics/population_prior_flipped.json").read_text())
    pop_feature_names = prior_flipped["feature_names"]
    coef_raw = np.array(prior_flipped["coefficients_population_std_units"]) / np.array(prior_flipped["population_scaler_scale"])
    name_to_idx = {n: i for i, n in enumerate(names_a)}
    X_36 = X_a[:, [name_to_idx[n] for n in pop_feature_names]]

    def _informed_flipped_fit_predict(X_tr, y_tr, X_te):
        scaler = StandardScaler()
        X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)
        fold_prior = coef_raw * scaler.scale_
        clf = InformedPriorLogisticRegression(prior=fold_prior, lam=5).fit(X_tr_s, y_tr)
        return clf.predict_proba(X_te_s)[:, 1]

    unique_days = sorted(set(groups_a))
    print(f"\nGenerating {N_BOOTSTRAP} shared day-draws (seed={SEED}) - one sequence, reused for every config ...")
    day_draws = build_shared_day_draws(unique_days, N_BOOTSTRAP, SEED)

    # Incremental save/resume: a kill here only costs the in-progress
    # candidate, not the whole run (this script has been killed by an
    # external time limit twice already at this data volume).
    results = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else {
        "day_draws_seed": SEED, "n_bootstrap": N_BOOTSTRAP, "unique_days": unique_days,
        "baseline_aucs": None, "candidates": {},
    }

    if results["baseline_aucs"] is None:
        print("\nComputing baseline (A/lr_l2_baseline) per-resample AUCs ...")
        baseline_aucs = replicate_aucs_for_draws(X_a, y_a, groups_a, _lr_fit_predict, day_draws)
        print(f"  {np.sum(~np.isnan(baseline_aucs))}/{N_BOOTSTRAP} valid")
        results["baseline_aucs"] = baseline_aucs.tolist()
        OUT_PATH.write_text(json.dumps(results, indent=2))
    else:
        print("\nBaseline already computed, loading from disk ...")
        baseline_aucs = np.array(results["baseline_aucs"])

    # Ordered cheapest-to-most-expensive (LR/NB/MAP fit in milliseconds;
    # RandomForest refit N_BOOTSTRAP x ~4 folds is the slow one) so
    # incremental saving accumulates the most results before hitting
    # whatever the slowest candidates cost, given repeated external kills.
    candidates = {
        "D/lr_l2_baseline":          (X_d, y_d, groups_d, _lr_fit_predict, "D"),
        "D/gaussian_nb":             (X_d, y_d, groups_d, _nb_fit_predict, "D"),
        "B/gaussian_nb":             (X_b, y_b, groups_b, _nb_fit_predict, "B"),
        "informed_prior_flipped_l5": (X_36, y_a, groups_a, _informed_flipped_fit_predict, "A_36feat"),
        "D/rf_shallow":              (X_d, y_d, groups_d, _rf_fit_predict, "D"),
        "B/rf_shallow":              (X_b, y_b, groups_b, _rf_fit_predict, "B"),
    }

    print("\n--- Paired diff (candidate - baseline) per candidate ---")
    for name, (X, y, groups, fit_fn, fs) in candidates.items():
        if name in results["candidates"]:
            print(f"\n{name}: already computed, skipping")
            continue

        print(f"\nComputing {name} per-resample AUCs ...")
        chunk_path = Path(__file__).parent / f"_partial_{name.replace('/', '_')}.npy"
        if chunk_path.exists():
            cand_aucs = np.load(chunk_path)
            n_done = int(np.sum(~np.isnan(cand_aucs)))
            print(f"  resuming from partial checkpoint: {n_done}/{N_BOOTSTRAP} already done")
        else:
            cand_aucs = np.full(N_BOOTSTRAP, np.nan)

        # Checkpoint every 20 resamples - RandomForest candidates in
        # particular have repeatedly exceeded the available runtime for a
        # full N_BOOTSTRAP pass; this bounds the cost of a kill to at most
        # one chunk's worth of refits instead of the whole candidate.
        CHUNK = 20
        remaining = [i for i in range(N_BOOTSTRAP) if np.isnan(cand_aucs[i])]
        for start in range(0, len(remaining), CHUNK):
            idxs = remaining[start:start + CHUNK]
            partial = replicate_aucs_for_draws(X, y, groups, fit_fn, [day_draws[i] for i in idxs])
            for j, i in enumerate(idxs):
                cand_aucs[i] = partial[j]
            np.save(chunk_path, cand_aucs)
            print(f"  {int(np.sum(~np.isnan(cand_aucs)))}/{N_BOOTSTRAP} done (checkpointed)")

        both_valid = ~np.isnan(baseline_aucs) & ~np.isnan(cand_aucs)
        n_paired = int(both_valid.sum())
        diffs = cand_aucs[both_valid] - baseline_aucs[both_valid]

        mean_diff = float(diffs.mean())
        ci_low, ci_high = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
        pct_beat = float((diffs > 0).mean() * 100)
        cross_population = fs != "A" and fs != "A_36feat"

        print(f"  n_paired={n_paired}/{N_BOOTSTRAP}  mean_diff={mean_diff:+.3f}"
              f"  95% CI=[{ci_low:+.3f}, {ci_high:+.3f}]  % beat baseline={pct_beat:.1f}%"
              f"  {'[CROSS-POPULATION - see caveat]' if cross_population else '[matched population]'}")

        results["candidates"][name] = {
            "feature_set": fs,
            "cross_population_caveat": cross_population,
            "n_paired": n_paired,
            "mean_diff": mean_diff,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "pct_resamples_beat_baseline": pct_beat,
            "candidate_aucs": cand_aucs.tolist(),
        }
        OUT_PATH.write_text(json.dumps(results, indent=2))
        chunk_path.unlink(missing_ok=True)
        print(f"  (saved)")

    print(f"\nAll done. Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
