# Clean-early-segment HRV/HR - results

Standalone experiment under `experiments/hrv_hr_early_window/`, kept
separate from `experiments/model_sweep/`'s feature sets A-D. Nothing
under `host/pipeline/`, `host/training/`, raw session data, `CLAUDE.md`,
`experiments/informed_prior/`, `experiments/diagnostics/`, or
`experiments/model_sweep/` was modified - reuses `compute_features`,
`collect_windows`, `windows_to_features`, and `day_based_cv` exactly as
they exist. The dual-window (60s + 180s) anchor collection and the
day-block bootstrap are reimplemented fresh here rather than imported from
other experiment folders, so this stays fully self-contained.

## Clean early segment definition

First 90 seconds of the 180s lookback window: `[end_ms-180000, end_ms-90000)`.
No pre-existing convention for this exists in the codebase (checked -
`host/pipeline/features.py`'s `_ppg_hr_features` has no early/late-segment
concept), so 90s was a reasonable default to start with.
That leaves a 90-second gap between the clean segment and the window's end
- which itself sits `buffer_ms=7000` before the actual event - so
**~97 seconds total gap to event onset**, well past the 15-30s minimum I
wanted to keep clear of motion artifact. `CLEAN_SEGMENT_MS` is a module constant in
`clean_segment_features.py` if you want to try a different split.

## Quality funnel - the headline finding, arguably more important than the AUCs

| Stage | n | % of dual-window anchors |
|---|---|---|
| Anchors with valid 60s **and** 180s windows | 202 | 100% |
| -> plausible mean HR (40-180bpm) in the clean segment | 195 | 96.5% |
| -> enough valid beats for HR trend | 195 | 96.5% |
| -> enough valid IBIs for HRV (RMSSD/SDNN) | 193 | 95.5% |
| **-> pass both gates (used below)** | **193** | **95.5%** |

**This is the opposite of what the motion-artifact hypothesis might have predicted, and it's good news**: only ~4.5% of windows had to be dropped for unusable cardiac signal. The clean-early-segment strategy appears to genuinely sidestep most of the motion-artifact problem that sank an earlier whole-window attempt (which got AUC 0.40-0.56, near chance). That prior failure mode was specifically attributed to motion during and immediately before the event - pulling the analysis window ~97s away from the event, as done here, seems to avoid most of it in practice, not just in theory.

## Sample-size caveat - real, but only partial

The common subset (193, anchors passing both quality gates) is **71.5%** of the full 60s-window baseline's size (270) - some anchors near session starts don't have the full 180s of prior data the large window needs. This is the same *kind* of population-mismatch concern flagged for feature sets B/D in `model_sweep`, though less severe (71.5% vs. B/D's ~65-70%, similar ballpark actually) - I bootstrapped anyway since it clears the 70% tolerance threshold, but it's worth being direct about what this means: **the common-subset population is measurably easier than the full population.** The full-reference baseline scores AUC 0.631+/-0.067; the *same* IMU-only model on the common subset alone scores 0.732+/-0.114 - a full 0.1 AUC higher, from population selection alone, before any HR/HRV features are added. Every comparison below is `A_only` vs. `A_plus_X`, both on the identical common subset - that comparison is fair - but none of these four numbers should be compared directly against the 0.631 full-baseline figure.

## Results

| Config | n | features | CV AUC | CV F1 | Bootstrap mean | Bootstrap 95% CI |
|---|---|---|---|---|---|---|
| Full reference (A, all valid anchors) | 270 | 54 | 0.631+/-0.067 | 0.604 | - | not applicable here |
| A_only (common subset) | 193 | 54 | 0.732+/-0.114 | 0.792 | 0.658 | [0.432, 0.810] |
| A + HR trend | 193 | 56 | 0.726+/-0.112 | 0.793 | 0.662 | [0.434, 0.808] |
| **A + HRV (RMSSD, SDNN)** | 193 | 56 | **0.761+/-0.077** | 0.803 | **0.701** | [0.476, 0.818] |
| A + both | 193 | 58 | 0.753+/-0.075 | 0.784 | 0.695 | [0.468, 0.818] |

(Bootstrap: 200 resamples, day-block, same procedure used throughout this project - resample days with replacement, refit per resample. Not a paired comparison against `A_only` - see note below.)

**HR trend adds nothing** - 0.726 vs. 0.732, if anything marginally worse, both in CV and bootstrap. Not worth carrying forward on its own.

**HRV is the interesting one, and the effect is consistent, not a fluke of one metric**: adding RMSSD/SDNN moves CV AUC from 0.732 -> 0.761 (+0.029) *and* tightens the CV std from 0.114 -> 0.077 *and* moves the bootstrap mean from 0.658 -> 0.701 (+0.043) *and* raises the bootstrap CI's lower bound from 0.432 -> 0.476. Every one of those four numbers moved the same direction. That's more consistent than most of what's shown up in this whole project's search so far.

**But the bootstrap CIs still overlap heavily** ([0.432, 0.810] vs. [0.476, 0.818]) - by the same standard applied everywhere else in this project, this is not a statistically confirmed improvement. I didn't run a *paired* bootstrap here initially (same day-draw sequence differenced against `A_only`, the way `model_sweep/paired_bootstrap.py` did) - given how much narrower that made the picture in `model_sweep` (revealing 77-95% per-resample win rates despite overlapping marginal CIs), it's the natural next step for a sharper answer than "overlapping but consistently favorable." (Follow-up below.)

## Honest read

**Worth pursuing further - this is the first PPG-derived signal in this entire project that hasn't just washed out.** Three independent things point the same direction: the signal-availability rate is high (95.5%, addressing the practical-reliability question directly), the HRV-specific improvement is consistent across CV and bootstrap and both point estimate and variance, and it's isolated cleanly to HRV rather than HR trend (which tracks with plausible physiology - HRV/RMSSD reflects autonomic/parasympathetic tone, a more specific signal than a raw HR level or slope). That's a meaningfully different, more encouraging result than the earlier whole-window attempt.

That said, don't oversell it: it's still not statistically confirmed at this sample size (n=193, day-based CV with only a handful of usable days), and the 71.5%-of-baseline population shift means part of the apparent lift could in principle be entangled with whatever makes the common subset easier in the first place - though the fact that `A_only` and `A_plus_hrv` are compared on the *identical* subset should mostly control for that. My recommendation: keep the clean-early-segment HRV features as a live candidate, don't merge them into `host/training/` yet, and prioritize either (a) a paired bootstrap for a sharper significance read, or (b) more recording days, which would help distinguish "real, modest effect" from "n=193 got lucky" - the same limiting factor behind every other result in this project.

## Follow-up: paired bootstrap, A_only vs. A_plus_hrv

The marginal CIs above overlapped heavily, but per `model_sweep/paired_bootstrap.py`'s own reasoning (used here as the template, same algorithm and reporting style), marginal CIs conflate shared "which days are hard" noise with the actual skill difference between two configs evaluated on the same days - pairing on the identical day-draw sequence cancels the shared part. Ran fresh in `paired_bootstrap.py` (`build_datasets.py`'s `build_all_configs()` reused unmodified - no per-resample arrays existed to pair post-hoc, same situation `model_sweep` hit).

**Population check, done programmatically rather than assumed**: before pairing, verified that `A_only` and `A_plus_hrv` share the literally identical row population - same sample count, elementwise-identical `y`, elementwise-identical `groups`, and `A_plus_hrv`'s first 54 columns exactly equal to `A_only`'s full feature matrix (it's `A_only` + exactly the 2 RMSSD/SDNN columns, nothing else differs). All checks passed. **This is a fully clean, matched-population pairing - no B/D-style cross-population caveat applies at all**, unlike most of `model_sweep`'s candidates.

One dataset-composition note: only 4 calendar days survive into this common subset (`2026-07-17`, `2026-07-22`, `2026-07-23`, `2026-07-24`) - `2026-07-16`'s sessions don't produce any anchors with a valid 180s window, so that day drops out entirely at the dual-window collection stage (it's present in the full 5-day/270-sample reference baseline, just not here).

**Result (500 resamples, both configs use only `LogisticRegression(C=0.01)` - no RandomForest bottleneck like model_sweep hit, so the full 500 ran directly, no reduction needed):**

| | Value |
|---|---|
| n_paired | 492 / 500 (8 dropped - insufficient unique days in that particular resample) |
| Mean paired diff (A_plus_hrv - A_only) | **+0.0426** |
| 95% CI of diff | **[+0.0045, +0.1173]** |
| % resamples A_plus_hrv beat A_only | **98.6%** |

**The CI excludes zero.** This is the first result across this entire investigation - `informed_prior/`, `diagnostics/`, `model_sweep/`, and this experiment - to formally cross the 95% significance threshold.

**But don't overread this - the margin is razor-thin.** The CI's lower bound is +0.0045: a hair above zero, not a comfortable margin. A slightly different resample seed, a few more or fewer samples, or a slightly different day landing in the data could plausibly pull that bound back below zero. This is "just barely significant," not "robustly, unambiguously significant." There's also a multiplicity concern worth naming honestly: across the full arc of this project, dozens of configs have been tested (46 in `model_sweep` alone, 8 lam values x 2 priors in `informed_prior`, several more in an earlier sweep and here). At a 5% significance threshold, finding roughly one "significant" result somewhere among that many tests is close to what you'd expect from chance alone even if nothing here were real. That doesn't mean this result *is* chance - it has a plausible physiological story (RMSSD/SDNN reflect autonomic tone, and this is the first time PPG-derived cardiac features were computed from a segment actually clean enough for that to show up) and a genuinely clean, fully-matched pairing behind it - but it means the honest confidence level here is "the most promising lead in the project so far," not "confirmed."

**Where this ranks against `informed_prior_flipped` (lam=5)**: this result is stronger, on both axes that matter.
- **Significance**: `informed_prior_flipped`'s CI was `[-0.010, +0.149]` - still included zero. This HRV result's CI is `[+0.0045, +0.1173]` - excludes it. HRV crosses the line `informed_prior_flipped` didn't quite reach.
- **Consistency**: 98.6% win rate here vs. 92.3% there.
- **Population cleanliness**: both are fully clean, matched-population pairings (`informed_prior_flipped` vs. baseline both ran on feature-set-A's identical population too) - so no advantage either way there.

Same tier of design (both clean pairings), but this result is the stronger evidence of the two - the first, and so far only, finding in this project to actually clear formal significance, even if only just.

## Honest verdict on this follow-up

**HRV's improvement holds up better than most of what's been tested in this project, and is the first result to formally clear the bar - but "clears the bar by a hair, after dozens of comparisons across the whole project" is a specific, calibrated level of confidence, not a green light.** Recommend: treat this as the top-priority candidate for further validation - ideally on new data that arrives *after* this analysis (a genuine prospective check, not another retrospective re-test on a dataset that keeps growing under the same tests), before it goes anywhere near `host/training/`.

## Files

- `clean_segment_features.py` - beat detection (reimplements `host/pipeline/features.py`'s bandpass+peak-detection methodology, since that module only exposes a fixed summary dict, not the peak positions this experiment needs), local-median ectopic-beat rejection, HR trend + HRV computation, per-window quality tracking
- `build_datasets.py` - feature-set-A reuse (exact, unmodified `collect_windows`/`windows_to_features`), dual-window (60s+180s) anchor collection, the 4 config assembly on the common quality-passing subset
- `paired_bootstrap.py` - paired day-block bootstrap follow-up (A_only vs. A_plus_hrv), template and algorithm from `model_sweep/paired_bootstrap.py`
- `paired_bootstrap_results.json` - raw per-resample paired results backing the table above
- `run_experiment.py` - `day_based_cv` (reused unmodified) on all 4 configs + reference baseline, conditional day-block bootstrap
- `results.json` - raw numeric outputs backing every table above
