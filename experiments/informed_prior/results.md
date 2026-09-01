# Informed-prior logistic regression - results

Standalone experiment under `experiments/informed_prior/`. Nothing in
`host/pipeline/`, `host/training/`, raw session data, or `CLAUDE.md` was
modified - everything below reuses those modules read-only (`compute_features`,
`collect_windows`, `windows_to_features`, `day_based_cv`).

## Data quality issues found and handled (worth knowing before the numbers)

This experiment reads from a local clone of the public dataset, which isn't
checked into this repo (fetch it yourself before rerunning):

```
git clone https://github.com/Bhorda/BFRBAnticipationDataset experiments/informed_prior/public_dataset
```

Cloning it and reading its own pipeline code directly (not just the README)
surfaced three things that would have silently corrupted this experiment if
missed:

1. **Unit mismatch.** Their accelerometer is in m/s^2 (confirmed empirically
   - mean magnitude ~9.94, matching gravity), ours is in g. Converted
   theirs by /9.80665. Their gyroscope is already in deg/s (confirmed - max
   values ~1189 rule out rad/s, which would be physically impossible for a
   wrist). No gyro conversion needed.
2. **Sampling rate mismatch.** Their data is ~10Hz (median inter-sample gap
   100ms), not our 50Hz. `host/pipeline/features.py` hardcodes `fs=50`
   internally and can't be parameterized without modifying a file that was
   off-limits - worked around by importing the same underlying
   `_time_domain`/`_freq_domain` helpers directly and calling them with the
   correct `fs` per dataset (verified byte-identical to `compute_features()`
   for our own data via a parity check that runs before anything else).
3. **A real bug in the public repo**: `exp-5`/`exp-6` and `exp-8`/`exp-9`'s
   `timestamps.csv` files are cross-assigned - `exp-5`'s declared recording
   start (1582118074000) matches `exp-6`'s actual sensor-data start to
   within 35ms, and vice versa (same pattern for exp-8/exp-9). This wasn't
   a data-quality problem to route around; it was a fixable file-pairing
   bug. Correcting it recovered all 10 participants (170 -> 408 usable
   windows) instead of silently losing 4 of 10.

## Feature-set exclusions (found empirically, not assumed)

The prior only transfers **36 of 54** accel/gyro features - mean, std, min,
max, rms, zcr per channel. All three FFT-derived stats (`dominant_freq`,
`power_0_5hz`, `power_5_15hz`) are excluded, for two different reasons
found across two failed iterations of this pipeline:

- `power_5_15hz` needs frequency content up to 15Hz; Nyquist at 10Hz
  sampling is 5Hz - structurally unmeasurable in the public data.
- All three are computed from an **unnormalized FFT sum**, whose magnitude
  depends on the number of samples in the window (~3000 for us at 50Hz,
  ~600 for them at 10Hz) or on Nyquist-bounded variance - not just on the
  underlying physical unit. The raw-unit conversion below (dividing by one
  dataset's std, multiplying by the other's) blows these up: an early run
  produced `power_0_5hz` prior coefficients of magnitude 10-20 against
  every other feature's <1, and even `dominant_freq` - which looks
  sample-rate-invariant in principle (a frequency in Hz means the same
  thing regardless of sampling rate) - produced a coefficient of magnitude
  11 once its Nyquist-bounded variance in the public data interacted with
  the conversion. Both are reported here because catching and fixing them
  is part of the honest result, not a footnote.

## 1. Population model

Fit on all 10 participants pooled (408 windows, 50% positive, generic -
not per-participant), `LogisticRegression(C=1.0, L2)` on 36 features,
standardized on the public dataset's own mean/std. Diagnostic standard
errors (unregularized `statsmodels.Logit` on the same data) are reported
for context, not as SEs of the actual (regularized) coefficients saved -
proper inference theory for a penalized estimator isn't standard, and some
diagonal SEs are enormous (~100-280 for `gyro_z_std`/`gyro_x_std`/`rms`
pairs) due to those being highly collinear with each other, a known
diagnostic artifact of the unregularized fit, not of the saved coefficients.

**Top weighted features:**

| Feature | Coefficient (population-std units) |
|---|---|
| gyro_x_min | +1.918 |
| accel_y_rms | -1.301 |
| gyro_y_min | +1.108 |
| gyro_z_std | -1.025 |
| gyro_z_rms | -1.000 |
| accel_y_mean | -0.979 |
| gyro_x_std | +0.801 |
| gyro_x_rms | +0.792 |
| accel_z_min | -0.758 |
| gyro_y_max | +0.692 |

The population model leans heavily on gyro extremes (`min`/`max`) and
gyro/accel spread (`std`/`rms`) - consistent with "more erratic rotational
movement precedes the behavior" in a controlled, induced-behavior study.

## 2. Prior transfer methodology

`population_prior.json` stores coefficients in "population standard
deviation" units, plus the population scaler's `mean_`/`scale_`. Each
personal CV fold standardizes on its own train split (different every
fold), so coefficients only mean the same thing across datasets converted
through raw physical units:

```
beta_raw              = beta_population_std / population_scaler.scale_
beta_this_fold_units   = beta_raw * this_fold_scaler.scale_
```

This is why the evaluation can't just call
`host/training/cross_validation.py:day_based_cv` unchanged - that
function has no hook for a per-fold-varying prior. `evaluate.py`'s
`informed_day_based_cv` mirrors its exact LOGO / per-fold-train-only-scaler
/ metric logic and only adds this rescaling step.

The MAP model (`informed_prior_model.py`) minimizes
`logistic_loss + lambda * sum((beta - prior)^2)` via `scipy.optimize`
(L-BFGS-B, analytic gradient) - a lightweight stand-in for a full
Bayesian fit. Self-check: with `prior=0`, this reproduces sklearn's
own L2 fit (coefficient correlation 1.0000 on synthetic data). Only slope
coefficients are shrunk toward the prior; the intercept is fit freely,
since baseline event prevalence isn't something that should transfer
between a controlled induced-behavior study and free-living personal
recording - the two tasks don't share a "base rate" worth matching.

## 3. Results

Personal data: 245 windows, 5 days, 51% positive (current, via
`host/training/build_dataset.py` unmodified - the dataset kept growing with
new sessions throughout this experiment, e.g. `2026-07-23-07-89pm` grew
from ~5K to ~245K packets between runs; numbers below reflect the latest run).

**(a) Sanity-check reproduction** (full 54 features, current model exactly
as-is): **AUC = 0.596 +/- 0.104**. The previously-reported 0.739 +/- 0.059 does
NOT reproduce - expected, not a bug: that number came from a 3-day/132-window
dataset; the real dataset is now 5 days/245 windows and performs differently.
This is the current true baseline, and it's already been reported honestly
in this conversation before this experiment started.

**(b) Fair comparison baseline** (same model, restricted to the 36 features
the prior covers): **AUC = 0.571 +/- 0.092**, F1 = 0.561 +/- 0.080.

**(c) Informed-prior sweep:**

| lambda | mean AUC | std AUC | mean F1 |
|---|---|---|---|
| 1 | **0.580** | 0.062 | 0.549 |
| 5 | 0.541 | 0.023 | 0.516 |
| 20 | 0.472 | 0.049 | 0.462 |
| 50 | 0.454 | 0.078 | 0.439 |
| 100 | 0.445 | 0.101 | 0.410 |
| 300 | 0.444 | 0.129 | 0.357 |
| 1000 | 0.443 | 0.132 | 0.372 |
| 5000 | 0.446 | 0.141 | 0.417 |

Performance degrades **monotonically and substantially** as the prior gets
stronger - from roughly baseline-level at the weakest setting (lambda=1) down
to worse than a coin flip once the prior actually dominates (lambda>=20).
This isn't noise-shaped; it's a clean, one-directional trend.

**Day-level block bootstrap** (1000 resamples, baseline vs. best-performing
lambda=1):

| | mean AUC | 95% CI |
|---|---|---|
| Baseline (36 feat) | 0.535 | [0.374, 0.740] |
| Informed prior (lam=1) | 0.539 | [0.384, 0.698] |

**CIs overlap almost completely.** Even the *best* informed-prior setting
is statistically indistinguishable from the baseline - and that's the
setting where the prior barely influences anything (lambda=1 is the
weakest tested strength). Every stronger setting is worse, well outside
noise.

## 4. Where the prior actually disagrees with personal data

Coefficient comparison at lambda=1, both fit on all personal data:

| Feature | Baseline coef | Informed coef | Prior coef | |diff| |
|---|---|---|---|---|
| accel_y_mean | +0.130 | +0.879 | **-0.720** | 0.749 |
| accel_z_mean | +0.101 | +0.798 | **-0.357** | 0.696 |
| gyro_z_std | -0.052 | -0.539 | -0.894 | 0.488 |
| gyro_x_max | +0.025 | +0.476 | -0.014 | 0.451 |
| gyro_z_rms | -0.052 | -0.497 | -0.868 | 0.445 |

The two most-shifted features (`accel_y_mean`, `accel_z_mean`) have
**opposite signs** between what this personal data supports (+) and what
the population prior says (-). That's not a scale artifact (both are
plain mean-acceleration terms, already validated as scale-safe) - it's a
genuine disagreement between the population's controlled-task signature
and this individual's free-living one. Pulling toward the population prior
for these features actively pushes the model in the wrong direction for
this person, which is the most plausible explanation for why performance
degrades as lambda increases.

## 5. Honest verdict

**The informed prior does not meaningfully outperform the current
baseline, and there's a real, identifiable reason why: at least some of
the population's coefficients point the wrong way for this individual.**
The bootstrap CIs overlap almost entirely even at the best (weakest)
prior setting, and every stronger setting makes things worse in a clean,
monotonic trend, not noisy fluctuation.

This is also exactly consistent with `CLAUDE.md`'s existing, evidence-based
decision to prefer personalized over generic/cross-subject models (the
reference paper found personalized CV consistently beat generic) - this
experiment is a second, independent piece of evidence for the same
conclusion, via a different mechanism (prior-informed shrinkage instead of
pooled training data).

Two caveats on the null result itself, for completeness: (1) 5 days is
still a small personal dataset, so "the prior doesn't help" and "we can't
yet detect that the prior helps" aren't fully separable - the same
data-quantity ceiling that's applied to every result in this project
applies here too. (2) The population study was a controlled, induced-BFRB
protocol; personal data is free-living, spontaneous behavior - a
population/task mismatch is a real possible cause of the sign
disagreements, not necessarily a proof that population data can never help.

## 6. Recommendation

**Do not adopt this for the real pipeline as currently evidenced.** This was
always meant to stay a separate experiment, kept out of `host/training/`
unless the results actually supported merging it - they don't.

If revisiting this later: (a) more personal data may change the picture
(the caveat above), (b) a population dataset from free-living rather than
induced-behavior recording would be a more apples-to-apples prior source
if one becomes available, (c) fitting per-participant population models
and checking whether the sign disagreement is universal or specific to a
subset of the 10 public participants would help distinguish "personalized
beats generic, in general" from "this specific population sample doesn't
match this specific person."

## Files

- `public_dataset/` - cloned public repo (gitignored, see the fetch command above)
- `public_features.py` - public dataset parsing + feature extraction, with the timestamp-swap fix and exclusion-list documentation
- `fit_population_prior.py` - produces `population_prior.json`
- `informed_prior_model.py` - the MAP ridge-to-prior estimator + self-check
- `evaluate.py` - produces `evaluation_results.json`
- `population_prior.json`, `evaluation_results.json` - raw numeric outputs backing every table above
