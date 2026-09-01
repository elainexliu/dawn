"""
bootstrap_informed_prior_flipped.py - targeted follow-up: the flipped
population prior (lam=5, its best point estimate at 0.651+/-0.096) landed
much higher than the original prior's best (0.580) and, unlike the
original, didn't degrade monotonically with lambda - a genuine surprise
relative to experiments/diagnostics/results.md's "mixed signal" verdict on
the sign-flip, which was based on coefficient-level comparison, not actual
CV performance. Worth a real bootstrap CI rather than just the point
estimate.

Usage:
    python -m experiments.model_sweep.bootstrap_informed_prior_flipped
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments.informed_prior.evaluate import day_block_bootstrap
from experiments.informed_prior.informed_prior_model import InformedPriorLogisticRegression
from experiments.model_sweep.feature_variants import FEATURE_SET_BUILDERS

BEST_LAM = 5
N_BOOTSTRAP = 500
OUT_PATH = Path(__file__).parent / "bootstrap_informed_prior_flipped.json"


def main():
    prior_flipped = json.loads(Path("experiments/diagnostics/population_prior_flipped.json").read_text())
    pop_feature_names = prior_flipped["feature_names"]
    coef_std = np.array(prior_flipped["coefficients_population_std_units"])
    scale = np.array(prior_flipped["population_scaler_scale"])
    coef_raw = coef_std / scale

    X_full, y, groups, names_full, ts = FEATURE_SET_BUILDERS["A"]()
    name_to_idx = {n: i for i, n in enumerate(names_full)}
    idx36 = [name_to_idx[n] for n in pop_feature_names]
    X_36 = X_full[:, idx36]

    def fit_predict(X_tr, y_tr, X_te):
        scaler = StandardScaler()
        X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)
        fold_prior = coef_raw * scaler.scale_
        clf = InformedPriorLogisticRegression(prior=fold_prior, lam=BEST_LAM).fit(X_tr_s, y_tr)
        return clf.predict_proba(X_te_s)[:, 1]

    print(f"Bootstrapping informed_prior_flipped (lam={BEST_LAM}), {N_BOOTSTRAP} resamples ...")
    boot = day_block_bootstrap(X_36, y, groups, fit_predict, n_bootstrap=N_BOOTSTRAP)
    print(f"  mean={boot['mean']:.3f}  CI=[{boot['ci_low']:.3f}, {boot['ci_high']:.3f}]"
          f"  (n_valid_replicates={boot['n_valid_replicates']}/{N_BOOTSTRAP})")

    OUT_PATH.write_text(json.dumps(boot, indent=2))
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
