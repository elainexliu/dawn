"""
bootstrap_top5.py - standalone continuation of run_sweep.py's final
ranking+bootstrap step, reusing the already-completed results_log.jsonl
(46 rows) instead of re-running the expensive nested ElasticNet search /
augmentation CV. Rebuilding the underlying datasets is
cheap (a few seconds each); refitting models 1000x per config is not, and
run_sweep.py's single monolithic process got killed partway through
bootstrapping (external time limit, not a code error - no traceback).

Saves incrementally to bootstrap_results.json after EACH config, so a kill
here loses at most one in-progress config, not the whole phase.

N_BOOTSTRAP reduced from 1000 to 500 for tractability given repeated
timeouts - noted explicitly in the report as a compute-driven reduction,
not a methodology change (still resamples days with replacement, refits
per resample, same procedure as experiments/informed_prior/evaluate.py's
day_block_bootstrap, reused unmodified).

Usage:
    python -m experiments.model_sweep.bootstrap_top5
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from experiments.informed_prior.evaluate import day_block_bootstrap
from experiments.model_sweep.feature_variants import FEATURE_SET_BUILDERS
from experiments.model_sweep.models import simple_classifiers
from experiments.model_sweep.sample_weighting import build_A_all_negatives, build_A_with_dirty_tags

OUT_DIR = Path(__file__).parent
LOG_PATH = OUT_DIR / "results_log.jsonl"
OUT_PATH = OUT_DIR / "bootstrap_results.json"
N_BOOTSTRAP = 500


def main():
    rows = [json.loads(l) for l in open(LOG_PATH)]
    valid = [r for r in rows if r.get("mean_auc") is not None and not np.isnan(r["mean_auc"]) and r["part"] != "sanity_check"]
    valid.sort(key=lambda r: -r["mean_auc"])
    top5 = valid[:5]
    print("Top 5 by mean AUC:")
    for r in top5:
        print(f"  {r['mean_auc']:.3f}+/-{r['std_auc']:.3f}  {r['feature_set']}/{r['model']}")

    print("\nRebuilding feature-set datasets (cheap - data loading only, no model fitting) ...")
    feature_datasets = {}
    for fs_name, builder in FEATURE_SET_BUILDERS.items():
        if any(r["feature_set"].replace("_36feat", "") == fs_name for r in top5) or fs_name == "A":
            X, y, groups, names, ts = builder()
            feature_datasets[fs_name] = (X, y, groups, names, ts)
            print(f"  {fs_name}: X={X.shape}")

    needs_dirty = any(r["model"] == "lr_l2_dirty_downweighted" for r in top5)
    needs_all_neg = any(r["model"] == "lr_l2_all_negatives_class_weighted" for r in top5)
    X_dirty = y_dirty = groups_dirty = None
    X_all_neg = y_all_neg = groups_all_neg = None
    if needs_dirty:
        print("  Rebuilding dirty-tagged dataset ...")
        X_dirty, y_dirty, groups_dirty, _, _ = build_A_with_dirty_tags()
    if needs_all_neg:
        print("  Rebuilding all-negatives dataset ...")
        X_all_neg, y_all_neg, groups_all_neg, _ = build_A_all_negatives()

    bootstrap_results = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else []
    already_done = {(b["feature_set"], b["model"]) for b in bootstrap_results}

    for r in top5:
        model = r["model"]
        fs_name = r["feature_set"].replace("_36feat", "")
        key = (r["feature_set"], model)
        if key in already_done:
            print(f"\n{r['feature_set']}/{model}: already bootstrapped, skipping")
            continue

        print(f"\nBootstrapping {r['feature_set']}/{model} ({N_BOOTSTRAP} resamples) ...")

        if model == "elasticnet_nested":
            print("  SKIPPED - would require re-running the full nested hyperparameter search"
                  f" inside each of {N_BOOTSTRAP} resamples (computationally infeasible here).")
            continue
        if model == "conditional_logit":
            print("  SKIPPED - fits on within-day matched-pair timestamp differences, which"
                  " day_block_bootstrap's (X, y, groups) interface has no hook for.")
            continue
        if model.startswith("lr_l2_augment"):
            print("  SKIPPED - augmentation happens on raw signal before feature extraction;"
                  " day_block_bootstrap only ever sees already-extracted features.")
            continue

        if model == "lr_l2_dirty_downweighted":
            X, y, groups = X_dirty, y_dirty, groups_dirty
            print("  NOTE: bootstrapped WITHOUT the dirty-downweighting itself (can't carry the"
                  " per-row weight array through day-based resampling) - treat as approximate.")
            def fit_predict(X_tr, y_tr, X_te):
                scaler = StandardScaler()
                X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)
                clf = LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced").fit(X_tr_s, y_tr)
                return clf.predict_proba(X_te_s)[:, 1]
        elif model == "lr_l2_all_negatives_class_weighted":
            X, y, groups = X_all_neg, y_all_neg, groups_all_neg
            def fit_predict(X_tr, y_tr, X_te):
                scaler = StandardScaler()
                X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)
                clf = LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced").fit(X_tr_s, y_tr)
                return clf.predict_proba(X_te_s)[:, 1]
        elif fs_name in feature_datasets:
            X, y, groups, names, ts = feature_datasets[fs_name]
            if model == "gaussian_nb":
                def fit_predict(X_tr, y_tr, X_te):
                    scaler = StandardScaler()
                    X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)
                    clf = GaussianNB().fit(X_tr_s, y_tr)
                    return clf.predict_proba(X_te_s)[:, 1]
            elif model in ("rf_shallow", "gbt_shallow"):
                clf_dict = simple_classifiers()
                def fit_predict(X_tr, y_tr, X_te, _clf=clf_dict[model]):
                    scaler = StandardScaler()
                    X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)
                    clf = clone(_clf).fit(X_tr_s, y_tr)
                    return clf.predict_proba(X_te_s)[:, 1]
            else:
                def fit_predict(X_tr, y_tr, X_te):
                    scaler = StandardScaler()
                    X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)
                    clf = LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced").fit(X_tr_s, y_tr)
                    return clf.predict_proba(X_te_s)[:, 1]
        else:
            print(f"  SKIPPED - no bootstrap fit_predict wired for this model type")
            continue

        boot = day_block_bootstrap(X, y, groups, fit_predict, n_bootstrap=N_BOOTSTRAP)
        boot["feature_set"] = r["feature_set"]
        boot["model"] = model
        boot["point_estimate_mean_auc"] = r["mean_auc"]
        bootstrap_results.append(boot)
        OUT_PATH.write_text(json.dumps(bootstrap_results, indent=2))
        print(f"  point={r['mean_auc']:.3f}  bootstrap mean={boot['mean']:.3f}"
              f"  CI=[{boot['ci_low']:.3f}, {boot['ci_high']:.3f}]  (saved)")

    # Baseline for comparison, if not already done
    if not any(b["feature_set"] == "A" and b["model"] == "lr_l2_baseline_REFERENCE" for b in bootstrap_results):
        print("\nBootstrapping the 0.596 baseline for reference ...")
        X_a, y_a, groups_a, names_a, ts_a = feature_datasets.get("A") or FEATURE_SET_BUILDERS["A"]()

        def _baseline_fit_predict(X_tr, y_tr, X_te):
            scaler = StandardScaler()
            X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)
            clf = LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced").fit(X_tr_s, y_tr)
            return clf.predict_proba(X_te_s)[:, 1]

        baseline_boot = day_block_bootstrap(X_a, y_a, groups_a, _baseline_fit_predict, n_bootstrap=N_BOOTSTRAP)
        baseline_boot["feature_set"] = "A"
        baseline_boot["model"] = "lr_l2_baseline_REFERENCE"
        bootstrap_results.append(baseline_boot)
        OUT_PATH.write_text(json.dumps(bootstrap_results, indent=2))
        print(f"  baseline: mean={baseline_boot['mean']:.3f}  CI=[{baseline_boot['ci_low']:.3f}, {baseline_boot['ci_high']:.3f}]")

    print(f"\nAll done. Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
