"""
run_experiment.py - stacks the three best leads so far to see if they
compound: model_sweep's B/D multi-horizon feature sets, thermal_and_buffer's
3s-buffer shortening, and hrv_hr_early_window's clean-segment RMSSD/SDNN.
Runs each stage through day-based LOGO CV, with a paired day-block bootstrap
between stages (7s->3s buffer, then no-hrv->+hrv on a matched subset).

Reuses feature_variants.py and clean_segment_features.py rather than
reimplementing them.

Usage:
    python -m experiments.best_model_followup.run_experiment
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from experiments.hrv_hr_early_window.clean_segment_features import (
    clean_segment_cardiac_features,
)
from experiments.model_sweep.feature_variants import (
    HORIZON_LENGTHS_MS,
    collect_multi_window_anchors,
    feature_set_B,
    feature_set_D,
)
from host.training.build_dataset import _clean_nans
from host.training.cross_validation import day_based_cv

OUT_PATH = Path(__file__).parent / "results.json"
N_BOOTSTRAP = 100
SEED = 42
FEATURE_SET_FNS = {"B": feature_set_B, "D": feature_set_D}
BUFFERS_MS = [7000, 3000]


def _rf_shallow():
    return RandomForestClassifier(
        n_estimators=200, max_depth=3, min_samples_leaf=8,
        class_weight="balanced", random_state=42,
    )


def _lr_l2_baseline():
    return LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced")


MODEL_CTOR = {"rf_shallow": _rf_shallow, "lr_l2_baseline": _lr_l2_baseline}


def _fit_predict(model_name, X_tr, y_tr, X_te):
    scaler = StandardScaler()
    X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)
    clf = MODEL_CTOR[model_name]().fit(X_tr_s, y_tr)
    return clf.predict_proba(X_te_s)[:, 1]


def build_shared_day_draws(unique_days, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    return [rng.choice(unique_days, size=len(unique_days), replace=True) for _ in range(n_bootstrap)]


def replicate_aucs_for_draws(X, y, groups, model_name, day_draws):
    """Same bootstrap logic as model_sweep/paired_bootstrap.py and
    thermal_and_buffer - duplicated rather than imported so each experiment
    folder stays self-contained."""
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
            y_prob = _fit_predict(model_name, X[train_idx], y_tr, X[test_idx])
            fold_aucs.append(roc_auc_score(y_te, y_prob))
        if fold_aucs:
            out[i] = float(np.mean(fold_aucs))
    return out


def paired_bootstrap(X_a, y_a, groups_a, X_b, y_b, groups_b, model_name, n_bootstrap, seed,
                      shared_days_source="a"):
    """Paired diff (b - a). If groups_a == groups_b (matched population),
    this is a clean pairing. Otherwise it's cross-population (pairs on
    shared day-draws only) - caller is responsible for flagging that."""
    days_source = groups_a if shared_days_source == "a" else groups_b
    unique_days = sorted(set(days_source))
    day_draws = build_shared_day_draws(unique_days, n_bootstrap, seed)
    aucs_a = replicate_aucs_for_draws(X_a, y_a, groups_a, model_name, day_draws)
    aucs_b = replicate_aucs_for_draws(X_b, y_b, groups_b, model_name, day_draws)
    both_valid = ~np.isnan(aucs_a) & ~np.isnan(aucs_b)
    diffs = aucs_b[both_valid] - aucs_a[both_valid]
    return {
        "n_paired": int(both_valid.sum()),
        "n_bootstrap": n_bootstrap,
        "mean_diff": float(diffs.mean()) if len(diffs) else float("nan"),
        "ci_low": float(np.percentile(diffs, 2.5)) if len(diffs) else float("nan"),
        "ci_high": float(np.percentile(diffs, 97.5)) if len(diffs) else float("nan"),
        "pct_resamples_favoring_b": float(100 * (diffs > 0).mean()) if len(diffs) else float("nan"),
    }


def build_feature_set(buffer_ms: int, which: str):
    fn = FEATURE_SET_FNS[which]
    anchors = collect_multi_window_anchors(buffer_ms=buffer_ms, lengths_ms=HORIZON_LENGTHS_MS)
    rows, labels, groups = [], [], []
    names = None
    for multi, label, day_str, end_ms in anchors:
        feats = fn(multi)
        if names is None:
            names = sorted(feats.keys())
        rows.append([feats[k] for k in names])
        labels.append(label)
        groups.append(day_str)
    X = np.array(rows, dtype=float)
    y = np.array(labels, dtype=int)
    groups_arr = np.array(groups)
    X, names = _clean_nans(X, names)
    return X, y, groups_arr, names, anchors


def add_hrv_matched(X, y, groups, anchors):
    """Restrict to anchors passing hrv_hr_early_window/'s HRV quality gate,
    then return (X_common_no_hrv, X_common_plus_hrv, y_common, groups_common)
    - a matched-population pair, same technique that experiment's own
    A_only vs. A_plus_hrv comparison used."""
    hrv_rows, ok_mask = [], []
    for multi, label, day_str, end_ms in anchors:
        res = clean_segment_cardiac_features(multi[180_000], end_ms)
        f = res["features"]
        hrv_rows.append([f["rmssd_clean"], f["sdnn_clean"]])
        ok_mask.append(bool(res["quality"]["hrv_ok"]))
    hrv_arr = np.array(hrv_rows, dtype=float)
    ok_mask = np.array(ok_mask)
    X_common = X[ok_mask]
    X_plus_hrv = np.hstack([X_common, hrv_arr[ok_mask]])
    return X_common, X_plus_hrv, y[ok_mask], groups[ok_mask], ok_mask


def main():
    results = {"cv": {}, "datasets_meta": {}, "bootstrap": {}}
    datasets = {}  # stage_key -> (X, y, groups)

    for which in ("B", "D"):
        for buffer_ms in BUFFERS_MS:
            print(f"\nBuilding feature set {which} @ buffer_ms={buffer_ms} ...")
            X, y, groups, names, anchors = build_feature_set(buffer_ms, which)
            stage_key = f"{which}@{buffer_ms}"
            datasets[stage_key] = (X, y, groups)
            results["datasets_meta"][stage_key] = {
                "n": int(X.shape[0]), "n_features": int(X.shape[1]),
                "positive_rate": float(y.mean()), "days": sorted(set(groups)),
            }
            print(f"  n={X.shape[0]}  features={X.shape[1]}  positive_rate={y.mean():.2%}  days={sorted(set(groups))}")

            for model_name in MODEL_CTOR:
                cv = day_based_cv(X, y, groups, MODEL_CTOR[model_name]())
                results["cv"].setdefault(stage_key, {})[model_name] = cv
                print(f"    {model_name}: AUC={cv['mean_auc']:.3f} +/- {cv['std_auc']:.3f}")

            if buffer_ms == 3000:
                X_common, X_hrv, y_c, groups_c, ok_mask = add_hrv_matched(X, y, groups, anchors)
                hrv_common_key = f"{which}@{buffer_ms}_common_no_hrv"
                hrv_key = f"{which}@{buffer_ms}+hrv"
                datasets[hrv_common_key] = (X_common, y_c, groups_c)
                datasets[hrv_key] = (X_hrv, y_c, groups_c)
                results["datasets_meta"][hrv_key] = {
                    "n": int(X_hrv.shape[0]), "n_features": int(X_hrv.shape[1]),
                    "positive_rate": float(y_c.mean()), "days": sorted(set(groups_c)),
                    "hrv_gate_pass_rate": float(ok_mask.mean()),
                }
                print(f"  HRV quality gate: {int(ok_mask.sum())}/{len(ok_mask)} anchors pass"
                      f" ({ok_mask.mean():.1%}) -> common-subset n={X_common.shape[0]}")

                for model_name in MODEL_CTOR:
                    cv_common = day_based_cv(X_common, y_c, groups_c, MODEL_CTOR[model_name]())
                    cv_hrv = day_based_cv(X_hrv, y_c, groups_c, MODEL_CTOR[model_name]())
                    results["cv"].setdefault(hrv_common_key, {})[model_name] = cv_common
                    results["cv"].setdefault(hrv_key, {})[model_name] = cv_hrv
                    print(f"    [common subset, no hrv] {model_name}: AUC={cv_common['mean_auc']:.3f} +/- {cv_common['std_auc']:.3f}")
                    print(f"    [common subset, +hrv]   {model_name}: AUC={cv_hrv['mean_auc']:.3f} +/- {cv_hrv['std_auc']:.3f}")

    print("\n=== Paired bootstrap: buffer_ms 7000 -> 3000 (cross-population, shared day-draws) ===")
    for which in ("B", "D"):
        X7, y7, g7 = datasets[f"{which}@7000"]
        X3, y3, g3 = datasets[f"{which}@3000"]
        for model_name in MODEL_CTOR:
            key = f"{which}_buffer_7000_to_3000_{model_name}"
            print(f"  {key} ...")
            r = paired_bootstrap(X7, y7, g7, X3, y3, g3, model_name, N_BOOTSTRAP, SEED, shared_days_source="a")
            r["cross_population"] = True
            results["bootstrap"][key] = r
            print(f"    mean_diff={r['mean_diff']:+.4f}  95% CI=[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
                  f"  % favoring 3000ms={r['pct_resamples_favoring_b']:.1f}%")
            OUT_PATH.write_text(json.dumps(results, indent=2))

    print("\n=== Paired bootstrap: 3000ms common subset, no-hrv -> +hrv (MATCHED population) ===")
    for which in ("B", "D"):
        Xc, yc, gc = datasets[f"{which}@3000_common_no_hrv"]
        Xh, yh, gh = datasets[f"{which}@3000+hrv"]
        for model_name in MODEL_CTOR:
            key = f"{which}_hrv_add_{model_name}"
            print(f"  {key} ...")
            r = paired_bootstrap(Xc, yc, gc, Xh, yh, gh, model_name, N_BOOTSTRAP, SEED, shared_days_source="a")
            r["cross_population"] = False
            results["bootstrap"][key] = r
            print(f"    mean_diff={r['mean_diff']:+.4f}  95% CI=[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
                  f"  % favoring +hrv={r['pct_resamples_favoring_b']:.1f}%")
            OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\nAll done. Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
