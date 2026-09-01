# Thermal features + pre-event buffer sweep - results

Standalone experiment under `experiments/thermal_and_buffer/`. `host/pipeline/`
and `host/training/` were imported read-only (`compute_features`,
`collect_windows`, `windows_to_features`/`_extract`, `build_dataset`,
`day_based_cv`) - nothing there was modified. Both scripts' pre-registered
plans are in their own module docstrings, written before results were seen.

Both experiments use `GradientBoostingClassifier(n_estimators=200, max_depth=3,
random_state=42)` as the primary model - the actual best model selected by
`host/training/train.py` on the current dataset, not a synthetic stand-in.

## 1. Does adding thermal features help?

`ambient_c` / `object_c` are parsed out of every raw packet already
(`build_dataset.py`'s `FIELDS_IMU_PPG`) but were never wired into
`compute_features()` - see that module's docstring, which only lists
accel/gyro/ppg_ir. This experiment adds 14 features (mean/std/min/max/rms/zcr
+ trend for each of `ambient_c`/`object_c`) and compares against the current
pipeline's feature set, **on the identical set of windows** - this is a
matched-population comparison (same 270 rows scored under both conditions),
the cleanest kind available in this project; no cross-population caveat
applies.

| Feature set | Model | Mean CV AUC | std |
|---|---|---|---|
| no thermal (current pipeline) | GBT | **0.641** | 0.038 |
| + thermal (14 new features) | GBT | 0.550 | 0.071 |
| no thermal (current pipeline) | LR (C=0.01) | 0.632 | 0.065 |
| + thermal (14 new features) | LR (C=0.01) | 0.569 | 0.065 |

**Paired day-block bootstrap (N=200, matched population, same day-draws both
sides):**

| Model | Mean diff (thermal - baseline) | 95% CI | % resamples favoring thermal |
|---|---|---|---|
| GBT | -0.029 | [-0.117, +0.108] | 26.8% |
| LR | -0.025 | [-0.135, +0.087] | 31.4% |

**Verdict: thermal features don't help, and lean toward hurting.** The CI
contains zero for both models, so this isn't a statistically confirmed
regression - but the point estimate drops by ~0.07-0.09 AUC in both models,
and only about a quarter to a third of bootstrap resamples favor keeping
thermal in. At 270 samples and only 58 baseline features, adding 14 more
(some, like `*_zcr` on a slowly-drifting temperature signal, that don't mean
much physically) is plausibly just adding noise dimensions for GBT/LR to
overfit day-to-day thermal drift (room temperature, sensor contact, time of
day) rather than a real precursor signal. Skin temperature likely also just
lags autonomic arousal more slowly than HR does, so it's a weak candidate
for the seconds-to-minutes anticipatory window this project predicts on.
**Not recommended for the current pipeline.**

## 2. Does the pre-event buffer (7s) matter?

`DEFAULT_BUFFER_MS = 7000` absorbs the fact that a self-reported keypress
lags the actual urge onset by an unknown amount (see `build_dataset.py`).
Tested 7s (current), 5s, and 3s - same 60s window length, only the anchor
point shifts. **Unlike the thermal comparison, this is NOT matched-population**:
changing the buffer changes which 60s of signal gets sampled for every
window, so sample rows differ across buffer values even though all three
span the same 5 calendar days (`2026-07-16/17/22/23/24`). Flagged
cross-population, same caveat class as `experiments/model_sweep/`'s B/D-vs-A
comparisons - pairing is on shared day-draws, not identical rows.

| Buffer | n | Positive rate | Mean CV AUC (GBT) | std |
|---|---|---|---|---|
| 7s (current) | 270 | 51.1% | 0.641 | 0.038 |
| 5s | 271 | 51.3% | 0.617 | 0.054 |
| 3s | 273 | 50.9% | **0.669** | 0.041 |

**Paired day-block bootstrap vs. 7s (N=100, reduced from the planned 200 -
3 dataset builds + 2 candidates' worth of GBT refits made the full N=200 run
exceed available runtime; same reduction precedent as
`experiments/model_sweep/paired_bootstrap.py`):**

| Buffer | Mean diff (vs. 7s) | 95% CI | % resamples favoring |
|---|---|---|---|
| 5s | +0.031 | [-0.046, +0.126] | 66.0% |
| 3s | +0.089 | [-0.050, +0.162] | **94.8%** |

**Verdict: 3s is a real lead, not yet a confirmed win.** Its point estimate
is the best of the three buffers tested, and 94.8% of paired resamples favor
it over the current 7s default - just short of the 95% significance line,
and the CI still technically spans zero. This is the same shape of result as
the cardiac-signal finding elsewhere in this project (strong directional
signal, CI not fully excluding zero) - worth specifically re-testing once
more days of data accumulate, and worth trying as a default in the next
retrain, but not yet a basis for changing `DEFAULT_BUFFER_MS` in
`host/training/build_dataset.py` on this evidence alone. 5s shows a much
weaker, less consistent edge (66% win rate, smaller point diff) and isn't
worth pursuing on its own.

## Files

- `thermal_experiment.py` - feature-set comparison, matched population
- `buffer_experiment.py` - buffer_ms sweep, cross-population comparison
- `thermal_results.json`, `buffer_results.json` - full CV + bootstrap output
  backing the tables above, including per-fold AUCs
