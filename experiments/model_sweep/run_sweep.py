"""
run_sweep.py - orchestrates the full sweep (feature sets x models, sample
weighting, augmentation), logging every result incrementally to
results_log.jsonl so a crash partway through doesn't lose completed work,
and producing the final ranked table + bootstrap CIs.

Sample weighting (Part 3) and augmentation (Part 4) are layered only onto
the best-performing feature-set/model config from Part 2, not
cross-multiplied against every combination - full cross-multiplication
would be computationally infeasible here, and runs against this project's
own principle of not running unbounded searches on a 5-day dataset (see
experiments/informed_prior/results.md).

Usage:
    python -m experiments.model_sweep.run_sweep
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from experiments.informed_prior.evaluate import day_block_bootstrap, informed_day_based_cv
from experiments.informed_prior.informed_prior_model import InformedPriorLogisticRegression
from experiments.model_sweep.augmentation import AUGMENTATIONS, augmented_day_cv
from experiments.model_sweep.feature_variants import (
    FEATURE_SET_BUILDERS,
    _select_accel_gyro,
    feature_set_A,
)
from experiments.model_sweep.models import (
    conditional_logit_day_cv,
    nested_elasticnet_day_cv,
    simple_classifiers,
)
from experiments.model_sweep.sample_weighting import (
    build_A_all_negatives,
    build_A_with_dirty_tags,
    dirty_sample_weights,
)
from host.training.build_dataset import collect_windows
from host.training.cross_validation import day_based_cv
from sklearn.preprocessing import StandardScaler

OUT_DIR = Path(__file__).parent
LOG_PATH = OUT_DIR / "results_log.jsonl"
LAM_SWEEP = [1, 5, 20, 50, 100, 300, 1000, 5000]
N_BOOTSTRAP = 1000

results: list[dict] = []


def log(record: dict):
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    results.append(record)
    with open(LOG_PATH, "a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    tag = f"{record.get('feature_set','')}/{record.get('model','')}"
    print(f"  [{tag:<30}] AUC={record['mean_auc']:.3f}+/-{record['std_auc']:.3f}  F1={record['mean_f1']:.3f}+/-{record['std_f1']:.3f}")


def main():
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    print("=== Sanity check: reproduce the 0.596 baseline ===")
    X_a, y_a, groups_a, names_a, ts_a = FEATURE_SET_BUILDERS["A"]()
    print(f"  Feature set A: X={X_a.shape}  positive_rate={y_a.mean():.2%}  days={sorted(set(groups_a))}")
    sanity_cv = day_based_cv(X_a, y_a, groups_a, LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced"))
    print(f"  AUC={sanity_cv['mean_auc']:.3f}+/-{sanity_cv['std_auc']:.3f}  (expected ~0.596+/-0.104 from experiments/informed_prior/)")
    log({"part": "sanity_check", "feature_set": "A", "model": "lr_l2_baseline", **sanity_cv})

    print("\n=== Part 1+2: feature sets x models ===")
    feature_datasets = {}
    for fs_name, builder in FEATURE_SET_BUILDERS.items():
        if fs_name == "A":
            X, y, groups, names, ts = X_a, y_a, groups_a, names_a, ts_a
        else:
            print(f"Building feature set {fs_name} ...")
            X, y, groups, names, ts = builder()
            print(f"  X={X.shape}  positive_rate={y.mean():.2%}")
        feature_datasets[fs_name] = (X, y, groups, names, ts)

        print(f"\n--- Feature set {fs_name} ---")
        for model_name, clf in simple_classifiers().items():
            if fs_name == "A" and model_name == "lr_l2_baseline":
                continue  # already logged as the sanity check
            cv = day_based_cv(X, y, groups, clf)
            log({"part": "part2", "feature_set": fs_name, "model": model_name, **cv})

        print(f"  running nested elasticnet CV (feature set {fs_name}) ...")
        en_cv, chosen_params = nested_elasticnet_day_cv(X, y, groups)
        log({"part": "part2", "feature_set": fs_name, "model": "elasticnet_nested", **en_cv,
             "chosen_params_per_fold": chosen_params})

        print(f"  running conditional logit (feature set {fs_name}) ...")
        cl_cv = conditional_logit_day_cv(X, y, groups, ts)
        log({"part": "part2", "feature_set": fs_name, "model": "conditional_logit", **cl_cv})

    print("\n=== Part 2 item 6: informed-prior MAP model (feature set A, 36-feature subset only) ===")
    print("  (Doesn't extend to feature sets B/C/D - the population prior has no coefficients")
    print("   for jerk or multi-horizon-prefixed features; applying it there would be meaningless.)")
    prior_orig = json.loads(Path("experiments/informed_prior/population_prior.json").read_text())
    # population_prior_flipped.json lives under experiments/diagnostics/
    # (the sign-flip test), not experiments/informed_prior/.
    prior_flipped = json.loads(Path("experiments/diagnostics/population_prior_flipped.json").read_text())
    pop_feature_names = prior_orig["feature_names"]
    assert pop_feature_names == prior_flipped["feature_names"]

    name_to_idx = {n: i for i, n in enumerate(names_a)}
    idx36 = [name_to_idx[n] for n in pop_feature_names]
    X_36 = X_a[:, idx36]

    for prior_label, prior_json in (("original", prior_orig), ("flipped", prior_flipped)):
        coef_std = np.array(prior_json["coefficients_population_std_units"])
        scale = np.array(prior_json["population_scaler_scale"])
        coef_raw = coef_std / scale
        for lam in LAM_SWEEP:
            cv = informed_day_based_cv(X_36, y_a, groups_a, coef_raw, lam)
            log({"part": "part2_informed_prior", "feature_set": "A_36feat", "model": f"informed_prior_{prior_label}",
                 "lam": lam, **cv})

    # --- Part 3: layered on the best Part 2 config ---
    print("\n=== Part 3: sample weighting, layered on the best Part 2 config ===")
    part2_rows = [r for r in results if r["part"] in ("part2", "sanity_check") and not np.isnan(r["mean_auc"])]
    best_part2 = max(part2_rows, key=lambda r: r["mean_auc"])
    print(f"  Best Part 2 config: {best_part2['feature_set']}/{best_part2['model']}  AUC={best_part2['mean_auc']:.3f}")
    print("  Part 3 variants use feature set A specifically (dirty-tagging and all-negatives")
    print("  collection are implemented for A; applying to B/C/D would require re-deriving the")
    print("  same session-level marker bookkeeping for multi-horizon windows, out of scope here).")

    print("  Building dirty/clean tagged dataset ...")
    X_dirty, y_dirty, groups_dirty, names_dirty, dirty_flags = build_A_with_dirty_tags()
    n_dirty = int(((y_dirty == 1) & dirty_flags).sum())
    print(f"  {n_dirty} of {int((y_dirty==1).sum())} positives flagged dirty")
    weights = dirty_sample_weights(y_dirty, dirty_flags)

    def _weighted_lr_cv(X, y, groups, weights):
        from sklearn.model_selection import LeaveOneGroupOut
        from sklearn.metrics import roc_auc_score, f1_score
        logo = LeaveOneGroupOut()
        aucs, f1s = [], []
        for train_idx, test_idx in logo.split(X, y, groups):
            y_te = y[test_idx]
            if len(np.unique(y_te)) < 2:
                continue
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X[train_idx])
            X_te_s = scaler.transform(X[test_idx])
            clf = LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced")
            clf.fit(X_tr_s, y[train_idx], sample_weight=weights[train_idx])
            probs = clf.predict_proba(X_te_s)[:, 1]
            preds = clf.predict(X_te_s)
            aucs.append(roc_auc_score(y_te, probs))
            f1s.append(f1_score(y_te, preds, zero_division=0))
        aucs_arr, f1s_arr = np.array(aucs), np.array(f1s)
        return {"mean_auc": float(aucs_arr.mean()), "std_auc": float(aucs_arr.std()),
                "mean_f1": float(f1s_arr.mean()), "std_f1": float(f1s_arr.std()),
                "fold_aucs": aucs_arr.tolist(), "fold_f1s": f1s_arr.tolist()}

    dirty_cv = _weighted_lr_cv(X_dirty, y_dirty, groups_dirty, weights)
    log({"part": "part3", "feature_set": "A", "model": "lr_l2_dirty_downweighted", **dirty_cv,
         "n_dirty_positives": n_dirty})

    print("  Building all-negatives (no subsampling) dataset ...")
    X_all_neg, y_all_neg, groups_all_neg, names_all_neg = build_A_all_negatives()
    print(f"  n={X_all_neg.shape[0]}  positive_rate={y_all_neg.mean():.2%}")
    all_neg_cv = day_based_cv(X_all_neg, y_all_neg, groups_all_neg,
                              LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced"))
    log({"part": "part3", "feature_set": "A", "model": "lr_l2_all_negatives_class_weighted", **all_neg_cv,
         "n_samples": int(X_all_neg.shape[0])})

    # --- Part 4: augmentation, layered on the best Part 2 config's feature set ---
    print("\n=== Part 4: raw-signal augmentation, layered on feature set A + LR L2 ===")
    windows_a = collect_windows()
    for aug_name in list(AUGMENTATIONS.keys()) + ["all_combined"]:
        aug_list = list(AUGMENTATIONS.keys()) if aug_name == "all_combined" else [aug_name]
        cv = augmented_day_cv(
            windows_a, lambda: LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced"),
            feature_set_A, augmentation_names=aug_list, n_augments_per_window=2,
        )
        log({"part": "part4", "feature_set": "A", "model": f"lr_l2_augment_{aug_name}", **cv})

    # --- Rank everything, bootstrap the top 5 ---
    print("\n=== Ranking all results, bootstrapping top 5 ===")
    valid = [r for r in results if not np.isnan(r["mean_auc"]) and r["part"] != "sanity_check"]
    valid.sort(key=lambda r: -r["mean_auc"])
    top5 = valid[:5]

    print("Top 5 by mean AUC (bootstrapping each) ...")
    bootstrap_results = []
    for r in top5:
        model = r["model"]
        fs_name = r["feature_set"].replace("_36feat", "")

        # Model types that don't cleanly fit day_block_bootstrap's
        # (X, y, groups, fit_predict_fn) interface - reported with an
        # explicit reason rather than silently substituted with a
        # different (wrong) model or a misleading generic fit.
        if model == "elasticnet_nested":
            print(f"  SKIPPING bootstrap for {fs_name}/{model} - would require re-running the full nested"
                  f" hyperparameter search inside each of {N_BOOTSTRAP} resamples (computationally infeasible"
                  f" here); point estimate should be treated with extra caution given no CI.")
            continue
        if model == "conditional_logit":
            print(f"  SKIPPING bootstrap for {fs_name}/{model} - conditional_logit_day_cv fits on within-day"
                  f" matched-pair timestamp differences, which day_block_bootstrap's (X, y, groups) interface"
                  f" (reused unmodified from experiments/informed_prior/) has no hook for passing through;"
                  f" point estimate should be treated with extra caution given no CI.")
            continue
        if model.startswith("lr_l2_augment"):
            print(f"  SKIPPING bootstrap for {fs_name}/{model} - augmentation happens on RAW signal before"
                  f" feature extraction, but day_block_bootstrap only ever sees already-extracted features;"
                  f" point estimate should be treated with extra caution given no CI.")
            continue

        if model == "lr_l2_dirty_downweighted":
            X, y, groups = X_dirty, y_dirty, groups_dirty
            def fit_predict(X_tr, y_tr, X_te):
                scaler = StandardScaler()
                X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)
                clf = LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced").fit(X_tr_s, y_tr)
                return clf.predict_proba(X_te_s)[:, 1]
            print(f"  NOTE: bootstrapping {fs_name}/{model} WITHOUT the dirty-downweighting itself"
                  f" (day_block_bootstrap's interface can't carry the extra per-row weight array through"
                  f" its day-resampling - see run_sweep.py); treat this bootstrap as approximate.")
        elif model == "lr_l2_all_negatives_class_weighted":
            X, y, groups = X_all_neg, y_all_neg, groups_all_neg
            def fit_predict(X_tr, y_tr, X_te):
                scaler = StandardScaler()
                X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)
                clf = LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced").fit(X_tr_s, y_tr)
                return clf.predict_proba(X_te_s)[:, 1]
        elif fs_name in feature_datasets and (model == "lr_l2_baseline" or model == "gaussian_nb" or model in ("rf_shallow", "gbt_shallow")):
            X, y, groups, names, ts = feature_datasets[fs_name]
            if model == "gaussian_nb":
                from sklearn.naive_bayes import GaussianNB
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
            print(f"  SKIPPING bootstrap for {fs_name}/{model} - no bootstrap fit_predict wired for this model type")
            continue

        boot = day_block_bootstrap(X, y, groups, fit_predict, n_bootstrap=N_BOOTSTRAP)
        boot["feature_set"] = r["feature_set"]
        boot["model"] = r["model"]
        boot["point_estimate_mean_auc"] = r["mean_auc"]
        bootstrap_results.append(boot)
        print(f"  {r['feature_set']}/{r['model']}: point={r['mean_auc']:.3f}  "
              f"bootstrap mean={boot['mean']:.3f}  CI=[{boot['ci_low']:.3f}, {boot['ci_high']:.3f}]")

    def _baseline_fit_predict(X_tr, y_tr, X_te):
        scaler = StandardScaler()
        X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)
        clf = LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced").fit(X_tr_s, y_tr)
        return clf.predict_proba(X_te_s)[:, 1]

    baseline_boot = day_block_bootstrap(X_a, y_a, groups_a, _baseline_fit_predict, n_bootstrap=N_BOOTSTRAP)
    print(f"  BASELINE (A/lr_l2_baseline) bootstrap: mean={baseline_boot['mean']:.3f}"
          f"  CI=[{baseline_boot['ci_low']:.3f}, {baseline_boot['ci_high']:.3f}]")

    summary = {
        "sanity_check": sanity_cv,
        "top5": top5,
        "bootstrap_top5": bootstrap_results,
        "bootstrap_baseline": baseline_boot,
        "best_part2_config": best_part2,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved: {OUT_DIR / 'summary.json'}")
    print(f"Full log: {LOG_PATH}")


if __name__ == "__main__":
    main()
