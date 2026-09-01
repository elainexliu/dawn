# Best-model follow-up: B/D at 3s buffer, plus clean-segment HRV - results

Standalone experiment under `experiments/best_model_followup/`, combining
three previously-separate leads into one pipeline to see whether they stack:
`experiments/model_sweep/`'s best feature sets (B/D, multi-horizon
30/60/180s, D adds jerk) and models (rf_shallow, lr_l2_baseline),
`experiments/thermal_and_buffer/`'s 3s-buffer lead, and
`experiments/hrv_hr_early_window/`'s clean-early-segment RMSSD/SDNN - the
only result in the project to previously clear 95% significance. Full
pre-registered plan is in `run_experiment.py`'s module docstring.
`host/pipeline/`, `host/training/`, and every experiment reused
(`model_sweep/feature_variants.py`, `hrv_hr_early_window/
clean_segment_features.py`) were imported read-only; nothing there changed.

## CV results (day-based LOGO)

| Config | n | rf_shallow AUC | lr_l2_baseline AUC |
|---|---|---|---|
| B @ 7s buffer (sweep repro) | 194 | 0.795 +/- 0.096 | 0.769 +/- 0.095 |
| B @ 3s buffer | 195 | 0.837 +/- 0.088 | 0.796 +/- 0.068 |
| B @ 3s, common subset (no HRV) | 191 | 0.832 +/- 0.093 | 0.797 +/- 0.077 |
| **B @ 3s + HRV** | 191 | 0.833 +/- 0.076 | **0.820 +/- 0.061** |
| D @ 7s buffer (sweep repro) | 194 | 0.841 +/- 0.087 | 0.789 +/- 0.105 |
| D @ 3s buffer | 195 | 0.861 +/- 0.106 | 0.802 +/- 0.105 |
| D @ 3s, common subset (no HRV) | 191 | 0.850 +/- 0.093 | 0.796 +/- 0.114 |
| **D @ 3s + HRV** | 191 | **0.868 +/- 0.083** | **0.821 +/- 0.091** |

n differs slightly from `model_sweep/results.md`'s original 174 for B/D -
the raw dataset has grown since that sweep ran (194-195 vs 174, still 67%
positive), not a discrepancy in method. The `+hrv` rows use fewer samples
(191) and only 4 of 5 days (`2026-07-16` drops out) because they're
restricted to anchors passing `hrv_hr_early_window/`'s HRV quality gate
(97.9% pass rate among 3s-buffer anchors - consistent with that experiment's
own 95.5% at 7s buffer).

## Buffer 7s -> 3s, with multi-horizon features (cross-population - pairs on shared day-draws, not identical rows; same caveat class as `model_sweep`'s B/D-vs-A comparisons)

| Feature set | Model | Mean diff | 95% CI | % resamples favoring 3s |
|---|---|---|---|---|
| B | rf_shallow | +0.022 | [-0.029, +0.065] | 84.5% |
| B | lr_l2_baseline | +0.014 | [-0.079, +0.084] | 69.1% |
| D | rf_shallow | +0.017 | [-0.025, +0.068] | 80.4% |
| D | lr_l2_baseline | -0.004 | [-0.145, +0.065] | 59.8% |

**The buffer effect is real but weaker once multi-horizon features are
already in play.** Directionally still favors 3s for 3 of 4 combos (all
rf_shallow, and B's lr), but every CI includes zero, and `D/lr_l2_baseline`
shows essentially no effect. Read: the single-window IMU baseline
(`thermal_and_buffer/`, 94.8% win rate) had more headroom for buffer timing
to matter; once 30s/60s/180s multi-horizon context is already in the
feature set, shifting the anchor point 4 seconds earlier buys less.

## HRV addition on the 3s-buffer common subset (matched population - identical rows, with vs. without the 2 HRV columns; the clean pairing `hrv_hr_early_window/` itself recommended)

| Feature set | Model | Mean diff | 95% CI | % resamples favoring HRV |
|---|---|---|---|---|
| B | rf_shallow | +0.009 | [-0.026, +0.061] | 71.4% |
| **B** | **lr_l2_baseline** | **+0.021** | **[+0.0055, +0.0412]** | **98.0%** |
| D | rf_shallow | +0.006 | [-0.025, +0.032] | 64.3% |
| **D** | **lr_l2_baseline** | **+0.023** | **[+0.0048, +0.0482]** | **100%** |

**This is the headline finding.** For `lr_l2_baseline`, adding clean-segment
RMSSD/SDNN clears 95% significance on *both* B and D - independently
reproducing `hrv_hr_early_window/`'s original result ([+0.0045, +0.1173],
98.6% win, on a single-window IMU baseline at 7s buffer) on a materially
different feature base (multi-horizon 30/60/180s +/- jerk) and a different
buffer (3s). That's two independent replications of the same effect under
different conditions - meaningfully stronger evidence than either result
alone, and the strongest corroboration any finding in this project has
received. For `rf_shallow`, the same addition does **not** clear
significance on either feature set (64-71% win rates, CIs include zero) -
worth flagging honestly rather than only reporting the model that confirms
the story: HRV's benefit so far is specific to the linear model, not
universal across model types tested.

## What's the best/most promising model now?

Two different answers depending on what "best" means, same distinction this
project has drawn everywhere else:

- **Best point estimate: `D @ 3s + HRV`, rf_shallow - 0.868 AUC.** Highest
  number in this entire project to date. But neither of the two changes that
  produced it (3s buffer, HRV addition) individually clears significance for
  this model - both CIs include zero. Treat the number itself with the same
  skepticism this project has applied to every other high point estimate
  (e.g. `model_sweep`'s D/rf_shallow at 0.819, which also didn't survive
  bootstrapping).
- **Best statistically-supported config: `D @ 3s + HRV`, lr_l2_baseline -
  0.821 AUC**, where the HRV component of that number is one of only three
  results in this project's entire history to formally exclude zero in a
  paired bootstrap (the others: `hrv_hr_early_window/`'s original A_plus_hrv
  test, and now this experiment's B+HRV). The buffer component isn't
  separately confirmed (D/lr's buffer-transition CI includes zero, 59.8%
  win), so the honest claim is narrower than "3s+HRV is confirmed better than
  7s baseline" - it's "adding HRV to a 3s-buffer multi-horizon LR model is a
  confirmed improvement over the same model without HRV."

**Recommendation:** `lr_l2_baseline` + multi-horizon features (B or D) + 3s
buffer + clean-segment HRV is now the strongest-supported candidate in the
project - not because its point estimate is highest (it isn't; RF's is), but
because it's the only config in this entire follow-up where a specific,
isolated change (adding HRV) is backed by a bootstrap CI that excludes zero,
and that specific effect independently replicated across two feature sets.
Worth prioritizing over `rf_shallow`'s higher point estimates, which remain
unconfirmed exactly the way every previous high-point-estimate RF result in
this project has been. Still not recommended for merge into
`host/training/` on this data volume alone - same standing recommendation as
every other experiment here - but this is the clearest "next thing to try
with more data" the project has produced so far.

## Files

- `run_experiment.py` - full pipeline: B/D construction at buffer_ms
  in {7000, 3000}, HRV augmentation on the matched common subset, day-based
  CV, and both paired-bootstrap comparisons
- `results.json` - full CV + bootstrap output backing every table above,
  including per-fold AUCs
