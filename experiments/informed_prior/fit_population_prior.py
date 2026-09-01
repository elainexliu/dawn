"""
fit_population_prior.py - fits a population-level reference logistic
regression on the public BFRBAnticipationDataset, pooling all 10
participants.

Standardized on the population dataset's own mean/std (self-contained, no
leakage from personal data), so coefficients come out in "population
standard deviation" units. evaluate.py converts these into whatever
standardization a given personal CV fold uses via raw physical units:
    beta_raw = beta_population_std / population_scaler.scale_
    beta_fold_units = beta_raw * personal_fold_scaler.scale_
That's why population_prior.json stores the population scaler's mean/scale
alongside the coefficients.

power_5_15hz features (6 of 54) are unmeasurable at this dataset's ~10Hz
sampling (Nyquist=5Hz) - see public_features.py - and are excluded from
the fit; evaluate.py falls back to a zero prior for these.

L2 fit at C=1.0 (population has 408 samples for 48 features, a much more
comfortable ratio than personal data's ~9:1). Standard errors come from a
separate unregularized statsmodels fit for diagnostic purposes only -
proper SEs for a penalized estimator aren't standard theory, so these
aren't the basis for the saved coefficients.

Usage:
    python -m experiments.informed_prior.fit_population_prior
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from experiments.informed_prior.public_features import UNRELIABLE_FEATURES, build_public_dataset, verify_parity

OUT_PATH = Path(__file__).parent / "population_prior.json"
SKLEARN_C = 1.0


def main():
    print("Verifying feature-extraction parity before fitting anything on public data ...")
    if not verify_parity():
        raise SystemExit("Parity check failed - aborting.")

    print("\nBuilding public dataset ...")
    X_full, y, groups, feature_names_full = build_public_dataset()
    print(f"Total: X={X_full.shape}  positive_rate={y.mean():.2%}  participants={sorted(set(groups))}")

    keep_mask = np.array([name not in UNRELIABLE_FEATURES for name in feature_names_full])
    X = X_full[:, keep_mask]
    feature_names = [n for n, k in zip(feature_names_full, keep_mask) if k]
    print(f"\nExcluding {len(UNRELIABLE_FEATURES)} power_5_15hz features (unmeasurable at ~10Hz sampling): "
          f"{UNRELIABLE_FEATURES}")
    print(f"Fitting on {X.shape[1]} features, {X.shape[0]} samples.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\nFitting sklearn LogisticRegression(C={SKLEARN_C}, L2) for point-estimate coefficients ...")
    clf = LogisticRegression(C=SKLEARN_C, max_iter=2000)
    clf.fit(X_scaled, y)
    coef = clf.coef_[0]
    intercept = float(clf.intercept_[0])

    print("Fitting statsmodels Logit (unregularized) on the same data for diagnostic SEs ...")
    X_sm = sm.add_constant(X_scaled)
    try:
        sm_model = sm.Logit(y, X_sm).fit(disp=0, maxiter=200)
        se = sm_model.bse[1:].tolist()  # drop the constant's SE
        sm_converged = True
        pvalues = sm_model.pvalues[1:].tolist()
    except Exception as e:
        print(f"  statsmodels fit did not converge cleanly ({e}) - SEs will be omitted, "
              f"coefficients above are unaffected (they come from the separate sklearn fit).")
        se = [float("nan")] * len(feature_names)
        pvalues = [float("nan")] * len(feature_names)
        sm_converged = False

    order = np.argsort(-np.abs(coef))
    print(f"\n--- Top 10 population coefficients by |weight| ---")
    for i in order[:10]:
        se_str = f"  SE(diag)={se[i]:.3f}" if sm_converged else ""
        print(f"  {feature_names[i]:<28} coef={coef[i]:+.4f}{se_str}")

    result = {
        "feature_names": feature_names,
        "excluded_features": list(UNRELIABLE_FEATURES),
        "coefficients_population_std_units": coef.tolist(),
        "intercept_population_std_units": intercept,
        "diagnostic_se_unregularized_logit": se,
        "diagnostic_pvalues_unregularized_logit": pvalues,
        "statsmodels_converged": sm_converged,
        "population_scaler_mean": scaler.mean_.tolist(),
        "population_scaler_scale": scaler.scale_.tolist(),
        "n_samples": int(X.shape[0]),
        "n_participants": len(set(groups)),
        "positive_rate": float(y.mean()),
        "sklearn_C": SKLEARN_C,
    }
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
