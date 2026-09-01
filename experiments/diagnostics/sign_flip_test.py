"""
sign_flip_test.py - does a 180-degree accel_y/accel_z mounting orientation
difference explain the sign disagreement found in informed_prior/results.md?

Duplicates public_features.py's build_public_dataset() loop with accel_y/
accel_z negated right after unit conversion, before feature extraction -
a full re-extraction rather than negating already-computed features, since
min/max need to swap-and-negate (min(-x) = -max(x)), not just flip sign.

Usage:
    python -m experiments.diagnostics.sign_flip_test
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from experiments.informed_prior.evaluate import (
    _load_personal_data,
    _restrict_to_population_features,
    day_block_bootstrap,
    informed_day_based_cv,
)
from experiments.informed_prior.public_features import (
    PUBLIC_DATA_DIR,
    UNRELIABLE_FEATURES,
    _load_participant,
    _window_features,
)

OUT_DIR = Path("experiments/diagnostics")
SKLEARN_C = 1.0
LAM_SWEEP = [1, 5, 20, 50, 100, 300, 1000, 5000]


def build_public_dataset_flipped() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Identical to public_features.build_public_dataset(), except
    accel_y/accel_z are negated (in physical units, post unit-conversion)
    before any feature is computed."""
    exp_dirs = sorted(PUBLIC_DATA_DIR.glob("exp-*"))
    feature_names: list[str] | None = None
    rows, labels, groups = [], [], []

    for exp_dir in exp_dirs:
        pid = exp_dir.name
        acc, gyr, events = _load_participant(exp_dir)
        acc = acc.copy()
        acc["accel_y"] = -acc["accel_y"]
        acc["accel_z"] = -acc["accel_z"]
        # gyro deliberately untouched - testing an accel-only mounting flip

        t_min = max(acc.index.min(), gyr.index.min())
        t_max = min(acc.index.max(), gyr.index.max())
        n_pos = 0
        event_starts = [e["start_ms"] for e in events]

        for e in events:
            feats = _window_features(acc, gyr, e["start_ms"])
            if feats is None:
                continue
            if feature_names is None:
                feature_names = sorted(feats.keys())
            rows.append([feats[k] for k in feature_names])
            labels.append(1)
            groups.append(pid)
            n_pos += 1

        rng = np.random.default_rng(hash(pid) % (2**32))
        candidates = []
        t = t_min + 60_000
        while t <= t_max:
            if not any(abs(t - es) < 60_000 for es in event_starts):
                candidates.append(t)
            t += 5000
        chosen = rng.choice(candidates, size=min(n_pos, len(candidates)), replace=False) if candidates else []
        for end_ms in sorted(int(c) for c in chosen):
            feats = _window_features(acc, gyr, end_ms)
            if feats is None:
                continue
            rows.append([feats[k] for k in feature_names])
            labels.append(0)
            groups.append(pid)

    X = np.array(rows, dtype=float)
    y = np.array(labels, dtype=int)
    return X, y, np.array(groups), feature_names


def fit_population_prior_flipped(X_full, y, groups, feature_names_full) -> dict:
    """Mirrors experiments/informed_prior/fit_population_prior.py's fitting
    logic exactly (same exclusions, same SKLEARN_C, same standardization
    approach) on the sign-flipped data, without modifying that file."""
    keep_mask = np.array([name not in UNRELIABLE_FEATURES for name in feature_names_full])
    X = X_full[:, keep_mask]
    feature_names = [n for n, k in zip(feature_names_full, keep_mask) if k]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(C=SKLEARN_C, max_iter=2000)
    clf.fit(X_scaled, y)

    return {
        "feature_names": feature_names,
        "excluded_features": list(UNRELIABLE_FEATURES),
        "coefficients_population_std_units": clf.coef_[0].tolist(),
        "intercept_population_std_units": float(clf.intercept_[0]),
        "population_scaler_mean": scaler.mean_.tolist(),
        "population_scaler_scale": scaler.scale_.tolist(),
        "n_samples": int(X.shape[0]),
        "n_participants": len(set(groups)),
        "positive_rate": float(y.mean()),
        "sklearn_C": SKLEARN_C,
    }


def main():
    print("Building sign-flipped public dataset (accel_y, accel_z negated; gyro untouched) ...")
    X_full, y, groups, feature_names_full = build_public_dataset_flipped()
    print(f"  X={X_full.shape}  positive_rate={y.mean():.2%}  participants={len(set(groups))}")

    prior_flipped = fit_population_prior_flipped(X_full, y, groups, feature_names_full)
    (OUT_DIR / "population_prior_flipped.json").write_text(json.dumps(prior_flipped, indent=2))
    print(f"  Saved: {OUT_DIR / 'population_prior_flipped.json'}")

    original_prior = json.loads(Path("experiments/informed_prior/population_prior.json").read_text())
    pop_feature_names = prior_flipped["feature_names"]
    assert pop_feature_names == original_prior["feature_names"], "feature order mismatch between flipped and original priors"

    coef_flipped_std = np.array(prior_flipped["coefficients_population_std_units"])
    scale_flipped = np.array(prior_flipped["population_scaler_scale"])
    coef_raw_flipped = coef_flipped_std / scale_flipped

    coef_orig_std = np.array(original_prior["coefficients_population_std_units"])
    scale_orig = np.array(original_prior["population_scaler_scale"])
    coef_raw_orig = coef_orig_std / scale_orig

    print("\nLoading personal data and restricting to the population's feature set ...")
    X_personal_full, y_personal, groups_personal, feature_names_personal = _load_personal_data()
    X_personal = _restrict_to_population_features(X_personal_full, feature_names_personal, pop_feature_names)

    scaler_full = StandardScaler()
    X_personal_scaled = scaler_full.fit_transform(X_personal)
    baseline_fit = LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced").fit(X_personal_scaled, y_personal)

    print("\n--- Coefficient comparison table (mirrors results.md section 4), flipped prior ---")
    print(f"{'feature':<28} {'baseline':>10} {'orig prior':>12} {'flipped prior':>14} {'|orig diff|':>12} {'|flip diff|':>12}  outcome")
    rows_out = []
    for i, name in enumerate(pop_feature_names):
        b_base = baseline_fit.coef_[0][i]
        prior_orig_fold_units = coef_raw_orig[i] * scaler_full.scale_[i]
        prior_flip_fold_units = coef_raw_flipped[i] * scaler_full.scale_[i]
        diff_orig = abs(prior_orig_fold_units - b_base)
        diff_flip = abs(prior_flip_fold_units - b_base)
        same_sign_orig = np.sign(b_base) == np.sign(prior_orig_fold_units)
        same_sign_flip = np.sign(b_base) == np.sign(prior_flip_fold_units)

        if diff_flip < diff_orig * 0.8:
            outcome = "IMPROVED"
        elif diff_flip > diff_orig * 1.2:
            outcome = "WORSENED"
        else:
            outcome = "~unchanged"
        print(f"{name:<28} {b_base:>+10.3f} {prior_orig_fold_units:>+12.3f} {prior_flip_fold_units:>+14.3f}"
              f" {diff_orig:>12.3f} {diff_flip:>12.3f}  {outcome}"
              f"{'  [accel_y/z]' if name.startswith(('accel_y', 'accel_z')) else ''}")
        rows_out.append({
            "feature": name, "baseline_coef": float(b_base),
            "prior_orig": float(prior_orig_fold_units), "prior_flipped": float(prior_flip_fold_units),
            "same_sign_orig": bool(same_sign_orig), "same_sign_flipped": bool(same_sign_flip),
            "abs_diff_orig": float(diff_orig), "abs_diff_flipped": float(diff_flip), "outcome": outcome,
        })

    n_improved = sum(1 for r in rows_out if r["outcome"] == "IMPROVED")
    n_worsened = sum(1 for r in rows_out if r["outcome"] == "WORSENED")
    accel_yz_improved = sum(1 for r in rows_out if r["outcome"] == "IMPROVED" and r["feature"].startswith(("accel_y", "accel_z")))
    non_accel_yz_worsened = sum(1 for r in rows_out if r["outcome"] == "WORSENED" and not r["feature"].startswith(("accel_y", "accel_z")))
    print(f"\nSummary: {n_improved} features improved, {n_worsened} worsened overall.")
    print(f"  Of the {sum(1 for n in pop_feature_names if n.startswith(('accel_y','accel_z')))} accel_y/accel_z features: {accel_yz_improved} improved.")
    print(f"  Of features NOT accel_y/accel_z (gyro, accel_x - should be unaffected by an accel_y/z-only flip): {non_accel_yz_worsened} worsened.")

    results = {"coefficient_comparison": rows_out, "n_improved": n_improved, "n_worsened": n_worsened,
               "accel_yz_improved": accel_yz_improved, "non_accel_yz_worsened": non_accel_yz_worsened}

    # Only worth the full lambda sweep if the flip actually meaningfully helps.
    meaningfully_helps = accel_yz_improved >= 1 and n_worsened <= 2
    print(f"\nFlip {'meaningfully helps' if meaningfully_helps else 'does not clearly help'} "
          f"- {'running' if meaningfully_helps else 'SKIPPING'} the full lambda sweep + bootstrap.")

    if meaningfully_helps:
        print(f"\n--- Lambda sweep with sign-corrected prior ---")
        min_samples_features = _restrict_to_population_features(X_personal_full, feature_names_personal, pop_feature_names)
        sweep = {}
        for lam in LAM_SWEEP:
            cv = informed_day_based_cv(min_samples_features, y_personal, groups_personal, coef_raw_flipped, lam)
            sweep[lam] = cv
            print(f"  lam={lam:<6}  AUC={cv['mean_auc']:.3f}+/-{cv['std_auc']:.3f}  F1={cv['mean_f1']:.3f}+/-{cv['std_f1']:.3f}")
        best_lam = max(sweep, key=lambda l: sweep[l]["mean_auc"])
        print(f"  Best lam: {best_lam}  AUC={sweep[best_lam]['mean_auc']:.3f}+/-{sweep[best_lam]['std_auc']:.3f}")

        def _flipped_fit_predict(X_tr, y_tr, X_te):
            from experiments.informed_prior.informed_prior_model import InformedPriorLogisticRegression
            scaler = StandardScaler()
            X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)
            fold_prior = coef_raw_flipped * scaler.scale_
            clf = InformedPriorLogisticRegression(prior=fold_prior, lam=best_lam).fit(X_tr_s, y_tr)
            return clf.predict_proba(X_te_s)[:, 1]

        boot = day_block_bootstrap(min_samples_features, y_personal, groups_personal, _flipped_fit_predict)
        print(f"  Bootstrap (flipped, lam={best_lam}): mean={boot['mean']:.3f}  95% CI=[{boot['ci_low']:.3f}, {boot['ci_high']:.3f}]")

        original_eval = json.loads(Path("experiments/informed_prior/evaluation_results.json").read_text())
        print(f"  For comparison, ORIGINAL (non-flipped) best lam={original_eval['best_lam']}:"
              f" AUC={original_eval['lam_sweep'][str(original_eval['best_lam'])]['mean_auc']:.3f}"
              f"  bootstrap mean={original_eval['bootstrap_informed']['mean']:.3f}"
              f"  CI=[{original_eval['bootstrap_informed']['ci_low']:.3f}, {original_eval['bootstrap_informed']['ci_high']:.3f}]")

        results["lambda_sweep_flipped"] = {str(k): v for k, v in sweep.items()}
        results["best_lam_flipped"] = best_lam
        results["bootstrap_flipped"] = boot
        results["original_for_comparison"] = {
            "best_lam": original_eval["best_lam"],
            "sweep_at_best_lam": original_eval["lam_sweep"][str(original_eval["best_lam"])],
            "bootstrap": original_eval["bootstrap_informed"],
        }

    (OUT_DIR / "task2_raw_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved: {OUT_DIR / 'task2_raw_results.json'}")


if __name__ == "__main__":
    main()
