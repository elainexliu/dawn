# Pre-registration: confirmatory test for clean-early-segment HRV

**Written:** 2026-07-27
**Status:** written before any of the qualifying new recording days (see Section 5) exist. This is a genuine pre-registration, not a retrospective writeup - no new data collected under this spec has been analyzed as of this date.

This document freezes the exact procedure that produced the retrospective result in `results.md` (paired bootstrap, `A_plus_hrv` vs. `A_only`: mean diff +0.0426, 95% CI [+0.0045, +0.1173], 98.6% of resamples favoring `A_plus_hrv`). Every value below is pulled directly from the current implementation, not redesigned or re-derived.

---

## 1. Feature definition

- **Lookback window**: 180,000 ms (180s), ending `buffer_ms = 7,000 ms` before the event/anchor timestamp. (`host/training/build_dataset.py:DEFAULT_BUFFER_MS`, `experiments/hrv_hr_early_window/build_datasets.py:LARGE_WINDOW_MS`)
- **Clean early segment**: the first `CLEAN_SEGMENT_MS = 90,000 ms` (90s) of that 180s window. In absolute terms: `[end_ms - 180,000, end_ms - 90,000)`. (`clean_segment_features.py:CLEAN_SEGMENT_MS`, line 31)
- **IMU baseline window** (feature set A): a separate, standard 60,000 ms window ending at the same anchor point. (`build_datasets.py:X_WINDOW_MS`)
- **HRV metrics computed, BOTH included in the winning config**: `rmssd_clean` and `sdnn_clean` - confirmed directly in `build_datasets.py` line 171: `X_hrv = hstack([rmssd_clean, sdnn_clean])`. Not RMSSD alone, not SDNN alone.
- **HR trend features exist in the codebase but are NOT part of the winning config** (`A_plus_hrv`) and are explicitly out of scope here - see Section 4.

## 2. Artifact rejection rule

Pulled directly from `clean_segment_features.py`:

- Beat detection: 2nd-order Butterworth bandpass, 0.7-3.5 Hz, on `ppg_ir`; peaks via `scipy.signal.find_peaks` with minimum distance `= fs * 60 / 220` samples (220 bpm ceiling), `fs = 50 Hz`. (lines 35-38, 56-71)
- Ectopic-beat rejection: an IBI is rejected if it differs from the **local median** (the 5 neighboring beats on each side, `LOCAL_MEDIAN_WINDOW = 5`) by more than **20%** (`REJECT_FRAC_THRESHOLD = 0.20`). (lines 45-46, 74-90)
- RMSSD is computed only from **adjacent** valid-IBI pairs (never diffs across a rejected gap). SDNN is computed from the full set of valid IBIs, adjacency not required. (lines 154-162)

## 3. Quality gates and sample restriction (exact, including a detail easy to miss)

- `MIN_VALID_BEATS_FOR_TREND = 5`, `MIN_VALID_IBIS_FOR_HRV = 10`, `PLAUSIBLE_HR_RANGE_BPM = (40, 180)`. (lines 51-53)
- **The sample used for `A_only` and `A_plus_hrv` alike is restricted to `trend_ok & hrv_ok`** (`build_datasets.py` line 161, `common_mask = trend_ok & hrv_ok`) - i.e. windows must pass BOTH the HR-trend gate and the HRV gate, even though `A_plus_hrv` doesn't use the trend features. This is not incidental - it's the exact sample the retrospective result was computed on. The confirmatory test must apply the identical `trend_ok & hrv_ok` mask, not `hrv_ok` alone.

## 4. Model - frozen, no variants

- **Feature set**: `A_plus_hrv` exactly as defined in `build_datasets.py` - the 54 accel/gyro features from `compute_features()` (60s window) + `rmssd_clean` + `sdnn_clean` (56 features total). Compared against `A_only` (the same 54 features alone).
- **Classifier**: `LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced")`, `StandardScaler` fit per training fold. (`paired_bootstrap.py:_lr_fit_predict`)
- **Validation**: day-based grouping throughout (`day_based_cv` / the day-block bootstrap in `paired_bootstrap.py`) - never a random split.
- **Nothing about this may change when the confirmatory test is run**: no different `C`, no different rejection threshold, no different segment length, no feature-selection step, no re-tuning of anything, regardless of how the new data looks.

### Explicitly OUT OF SCOPE for the confirmatory test

Do not run any of the following as part of, or instead of, the confirmatory test:

- RMSSD-alone or SDNN-alone as separate configs
- `A_plus_hr_trend` or `A_plus_both` (both already exist in `build_datasets.py` but are not the winning config)
- Any alternate clean-segment length or position (e.g. first 60s, last 30s, a different offset)
- Any alternate rejection threshold (e.g. 15% or 25% instead of 20%) or local-median window size
- Any alternate classifier or hyperparameter (e.g. a different `C`, elastic net, RF)
- Any alternate lookback window length (e.g. 120s or 240s instead of 180s)

If any of these turn out to look interesting on the new data, that is a **new, separate, clearly-labeled exploratory analysis** - not the confirmatory test, and must not be reported as if it were.

## 5. Minimum new data before running the confirmatory test

- At least **5-8 new independent recording days**, none of which were part of the dataset this pre-registration was written against (`2026-07-16/17/22/23/24` - see `paired_bootstrap_results.json:unique_days` for the exact set at freeze time).
- Run the confirmatory test **once**, at a chosen point after that minimum is met - not repeatedly as data trickles in. Re-running it every time a new session lands recreates the exact multiple-comparisons problem this document exists to prevent. Pick a point, commit, run once.

## 6. Success criterion - stated in advance

- **Pass**: the 95% paired-bootstrap CI of `(A_plus_hrv AUC - A_only AUC)`, computed via the identical procedure in `paired_bootstrap.py`, excludes zero (entirely on the positive side) - the same bar the retrospective result cleared.
- **Fail**: the CI touches or includes zero. This is the honest answer if it happens, full stop. No post-hoc reinterpretation (e.g. "well the point estimate was still positive," "maybe with a different segment length," "the win rate was still high") - a CI that includes zero means the retrospective finding did not replicate.

## 7. Provenance

- Retrospective result this pre-registration is pinned to: `experiments/hrv_hr_early_window/paired_bootstrap_results.json`, computed on the 193-sample common subset spanning days `2026-07-17, 2026-07-22, 2026-07-23, 2026-07-24` (full reference baseline: 270 samples, 5 days, including `2026-07-16`, which drops out of the HRV common subset - see `results.md`).
- This document does not re-argue whether that retrospective result is promising - see `results.md` for that discussion. This document's only job is to pin down what gets tested against new data, exactly, and when.
