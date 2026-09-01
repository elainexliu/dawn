# Model sweep - results

Standalone experiment under `experiments/model_sweep/`. Nothing under
`host/pipeline/`, `host/training/`, raw session data, `CLAUDE.md`,
`experiments/informed_prior/`, or `experiments/diagnostics/` was modified -
everything here imports read-only from those (`compute_features`,
`collect_windows`, `windows_to_features`, `day_based_cv`,
`InformedPriorLogisticRegression`, `day_block_bootstrap`,
`population_prior.json`).

**Sanity check passed**: feature set A + `LogisticRegression(C=0.01)` via
day-based LOGO reproduces **AUC = 0.596 +/- 0.104** exactly, matching
`experiments/informed_prior/`'s verified baseline. Everything below is
trustworthy relative to that anchor.

**One correction made along the way**: I initially expected
`population_prior_flipped.json` under `experiments/informed_prior/`, but
it actually lives under `experiments/diagnostics/` (where the sign-flip
test ran). Fixed the path.

## Full results table (46 configs, sorted by mean AUC)

| AUC | std | F1 | Part | Feature set | Model |
|---|---|---|---|---|---|
| **0.819** | 0.105 | 0.849 | Part 2 | D | rf_shallow |
| 0.788 | 0.113 | 0.779 | Part 2 | D | lr_l2_baseline |
| 0.763 | 0.094 | 0.821 | Part 2 | B | rf_shallow |
| 0.749 | 0.093 | 0.794 | Part 2 | D | gaussian_nb |
| 0.741 | 0.109 | 0.786 | Part 2 | B | gaussian_nb |
| 0.740 | 0.102 | 0.775 | Part 2 | D | elasticnet_nested |
| 0.737 | 0.090 | 0.774 | Part 2 | B | lr_l2_baseline |
| 0.720 | 0.077 | 0.826 | Part 2 | D | gbt_shallow |
| 0.691 | 0.104 | 0.730 | Part 2 | B | elasticnet_nested |
| 0.687 | 0.105 | 0.682 | Part 2 | D | conditional_logit |
| 0.685 | 0.093 | 0.767 | Part 2 | B | gbt_shallow |
| 0.676 | 0.078 | 0.656 | Part 2 | B | conditional_logit |
| 0.664 | 0.128 | 0.598 | Part 2 | C | lr_l2_baseline |
| 0.664 | 0.110 | 0.656 | Part 2 | C | gaussian_nb |
| 0.656 | 0.125 | 0.496 | Part 3 | A | lr_l2_all_negatives_class_weighted |
| 0.651 | 0.112 | 0.651 | Part 2 | A | gaussian_nb |
| 0.651 | 0.096 | 0.628 | Part 2 item 6 | A (36 feat) | informed_prior_**flipped**, lam=5 |
| 0.648 | 0.129 | 0.580 | Part 2 | C | elasticnet_nested |
| 0.646 | 0.094 | 0.584 | Part 2 item 6 | A (36 feat) | informed_prior_flipped, lam=20 |
| 0.632 | 0.095 | 0.638 | Part 2 | C | gbt_shallow |
| 0.631 | 0.099 | 0.632 | Part 2 item 6 | A (36 feat) | informed_prior_flipped, lam=50 |
| 0.628 | 0.079 | 0.607 | Part 2 item 6 | A (36 feat) | informed_prior_flipped, lam=1 |
| 0.616 | 0.090 | 0.603 | Part 2 item 6 | A (36 feat) | informed_prior_flipped, lam=100 |
| 0.608 | 0.091 | 0.588 | Part 2 item 6 | A (36 feat) | informed_prior_flipped, lam=5000 |
| 0.607 | 0.092 | 0.591 | Part 2 item 6 | A (36 feat) | informed_prior_flipped, lam=300 |
| 0.607 | 0.090 | 0.604 | Part 4 | A | augment: all_combined |
| 0.606 | 0.093 | 0.585 | Part 2 item 6 | A (36 feat) | informed_prior_flipped, lam=1000 |
| 0.606 | 0.101 | 0.585 | Part 4 | A | augment: magnitude_warp |
| 0.605 | 0.102 | 0.582 | Part 4 | A | augment: jitter |
| 0.598 | 0.091 | 0.584 | Part 4 | A | augment: time_warp |
| **0.596** | **0.104** | 0.596 | **Reference** | **A** | **lr_l2_baseline (current pipeline, exactly as-is)** |
| 0.594 | 0.107 | 0.599 | Part 2 | C | rf_shallow |
| 0.584 | 0.108 | 0.611 | Part 3 | A | lr_l2_dirty_downweighted |
| 0.581 | 0.064 | 0.603 | Part 2 | A | gbt_shallow |
| 0.580 | 0.092 | 0.564 | Part 2 | A | elasticnet_nested |
| 0.580 | 0.062 | 0.549 | Part 2 item 6 | A (36 feat) | informed_prior_**original**, lam=1 (its best) |
| 0.555 | 0.052 | 0.561 | Part 2 | A | rf_shallow |
| 0.541 | 0.023 | 0.516 | Part 2 item 6 | A (36 feat) | informed_prior_original, lam=5 |
| 0.504 | 0.102 | 0.459 | Part 2 | A | conditional_logit |
| 0.503 | 0.100 | 0.454 | Part 2 | C | conditional_logit |
| 0.472 | 0.049 | 0.462 | Part 2 item 6 | A (36 feat) | informed_prior_original, lam=20 |
| 0.454 | 0.078 | 0.439 | Part 2 item 6 | A (36 feat) | informed_prior_original, lam=50 |
| 0.446 | 0.141 | 0.417 | Part 2 item 6 | A (36 feat) | informed_prior_original, lam=5000 |
| 0.445 | 0.101 | 0.410 | Part 2 item 6 | A (36 feat) | informed_prior_original, lam=100 |
| 0.444 | 0.129 | 0.357 | Part 2 item 6 | A (36 feat) | informed_prior_original, lam=300 |
| 0.443 | 0.132 | 0.372 | Part 2 item 6 | A (36 feat) | informed_prior_original, lam=1000 |

**Critical caveat on feature sets B and D**: they only have **174 samples at 67% positive**, vs. A/C's **245 samples at 51% positive** - the 180-second window required for the longest horizon can't be satisfied near the start of shorter sessions, and that survivorship isn't symmetric between classes. B and D's AUCs are **not computed on the same evaluation population as the 0.596 baseline** - part of their apparent edge could be the easier, more class-imbalanced sample rather than pure feature-engineering gain. This is exactly why bootstrap CIs matter more than the point estimates here.

**Nested ElasticNet's inner search was unstable**, worth flagging honestly: the chosen `(l1_ratio, C)` per outer fold on feature set A was `[(0.0,1.0), (1.0,0.1), (0.0,0.1), (0.0,1.0)]` - flip-flopping between pure-L2 and pure-L1 fold to fold. With only 3-4 days available inside each outer fold's inner search, there isn't enough data to pick a stable hyperparameter; elasticnet's mid-table finish should be read as "couldn't reliably tune," not "L1/L2 mixing doesn't help."

## Bootstrap CI overlap check (top 5 + baseline + the informed-prior surprise)

500-resample day-level block bootstrap (reduced from an initial 1000 for tractability after two runs took too long - same procedure, `day_block_bootstrap`, reused unmodified from `experiments/informed_prior/evaluate.py`):

| Config | Point AUC | Bootstrap mean | 95% CI |
|---|---|---|---|
| D/rf_shallow | 0.819 | 0.753 | [0.539, 0.845] |
| D/lr_l2_baseline | 0.788 | 0.726 | [0.518, 0.871] |
| B/rf_shallow | 0.763 | 0.718 | [0.552, 0.837] |
| D/gaussian_nb | 0.749 | 0.618 | [0.465, 0.815] |
| B/gaussian_nb | 0.741 | 0.611 | [0.455, 0.807] |
| informed_prior_flipped, lam=5 | 0.651 | 0.607 | [0.454, 0.773] |
| **Baseline** (A/lr_l2_baseline) | **0.596** | **0.548** | **[0.385, 0.772]** |

**Every single one overlaps the baseline's CI.** D/rf_shallow's point estimate (0.819) looks like a massive jump from 0.596, but its bootstrap CI's lower bound (0.539) sits well inside the baseline's CI, and the baseline's upper bound (0.772) sits well inside D/rf_shallow's CI. Same story for all six rows checked. **No config in this entire sweep clears the bar for a statistically distinguishable improvement over the current baseline, at this sample size.** This is the same conclusion every experiment in this project has reached - it's not a new failure, it's the same data-quantity ceiling showing up again, now confirmed against a much wider set of things that could plausibly have escaped it.

Elasticnet, conditional_logit, and the Part 4 augmentation variants weren't bootstrapped - documented reasons, not oversights:
- **elasticnet_nested**: bootstrapping properly would mean re-running the full nested grid search inside each of 500 resamples - computationally infeasible here.
- **conditional_logit**: fits on within-day matched-pair timestamp differences; `day_block_bootstrap`'s `(X, y, groups)` interface has no hook to carry timestamps through without modifying that reused function.
- **augmentation variants**: augmentation happens on raw signal before feature extraction, but `day_block_bootstrap` only ever sees already-extracted features.

Given none of the bootstrapped configs beat baseline, and augmentation's own point estimates (0.598-0.607) are already close to baseline (0.596) well within one std, it's very unlikely any of these three unbootstrapped categories would have changed the overall verdict.

## Where the informed-prior variants land

This directly updates `experiments/diagnostics/results.md`'s conclusion, and the update goes in an unexpected direction:

- **Original population prior**: consistent with everything found before. Best at lam=1 (0.580), degrades **monotonically** down to 0.443 at lam=1000 - same pattern as `experiments/informed_prior/results.md`. No contradiction here.
- **Flipped population prior** (the one `experiments/diagnostics/` called "mixed signal, doesn't confirm the orientation hypothesis" based on coefficient-level comparison): performs **substantially and consistently better than the original at every single lam tested** - 0.607-0.651 across the whole range, essentially flat instead of degrading. That flatness itself is notable: it means the flipped prior doesn't actively fight the personal data's signal the way the original does, regardless of how hard it's enforced.

That's a real, reproducible difference in actual predictive behavior between the two priors - which sits in tension with `diagnostics/results.md`'s coefficient-level "mixed signal" verdict (15/36 features improved, 14/36 worsened). Both findings can be true at once: individual coefficients can look noisy and mixed while the *aggregate* effect on a held-out prediction is still cleanly better. Worth flagging rather than glossing over. But it doesn't change the bottom line: bootstrapped, the flipped prior's best config (lam=5) still overlaps the baseline's CI entirely ([0.454, 0.773] vs. [0.385, 0.772]). **Interesting and worth remembering, not yet a confirmed win.**

## Recommendation

**No config from this sweep should be merged into `host/training/` or `CLAUDE.md`.** Every bootstrapped candidate - including the two that looked most dramatic in point estimate (D/rf_shallow at 0.819, D/lr_l2_baseline at 0.788) - has a confidence interval that fully overlaps the current 0.596 +/- 0.104 baseline's. Declaring a winner here would mean picking whichever config got luckiest on this specific bootstrap draw, which is precisely the overfitting-via-multiple-comparisons this sweep was designed to guard against, not fall into.

Two things worth carrying forward as *directions*, not conclusions:

1. **Multi-horizon feature engineering (feature sets B/D) consistently dominates the top of the point-estimate ranking** across every model type tested on top of it. That's suggestive, but confounded by B/D's smaller, differently-balanced sample (174 @ 67% positive vs. 245 @ 51%) - worth re-testing once there's enough data that the 180s-window coverage requirement doesn't force a meaningfully different evaluation population.
2. **The flipped population prior behaves genuinely differently from the original** - flat across lam instead of degrading - even though it doesn't yet beat baseline. Worth keeping as a candidate to revisit with more personal data, rather than discarding it entirely, since `diagnostics/results.md`'s own conclusion undersold how much the flip actually changes behavior.

Everything else - ElasticNet, conditional logit, Gaussian NB, shallow RF/GBT on the standard window, dirty-positive downweighting, all-negatives class-weighting, and all three raw-signal augmentation types - landed at or below baseline, within noise. None of it is a lever at this data volume.

## Follow-up: paired bootstrap (marginal CIs are a weaker test)

The marginal CIs above (baseline vs. each candidate, computed independently) all overlapped - but marginal CIs conflate two sources of noise: "which days happened to be easy or hard" (shared across every config, since they're all evaluated on the same days) and the actual candidate-vs-baseline skill difference. A **paired** bootstrap - the identical sequence of 200 day-block resamples applied to baseline and every candidate, then differenced resample-by-resample - cancels the shared noise and isolates the second part. This is a strictly stronger test, run fresh in `paired_bootstrap.py` since neither `bootstrap_results.json` nor `bootstrap_informed_prior_flipped.json` stored per-resample arrays to pair post-hoc (`day_block_bootstrap`, reused unmodified, only ever returns the aggregate).

**Pairing validity caveat, addressed rather than glossed over**: feature sets B/D have a different, smaller sample population than baseline's A (see the main caveat above - 180s-window coverage requirement). A, B, and D all span the identical 5 calendar days, so "apply the same sequence of which-days-got-drawn" is still well-defined across them - but it pairs on *shared day-level noise*, not on *identical rows*. That's a real, if partial, form of pairing (it should still cancel "day 3 was unusually easy for everyone," which is genuine shared noise), not a strict matched-pairs design. Flagged per-row below via `cross_pop=True`. The one comparison with **no such caveat** is `informed_prior_flipped` vs. baseline - both run on feature set A's exact same 260-row population, so that pairing is completely clean.

| Candidate | Mean paired diff | 95% CI of diff | % resamples beating baseline | Population match |
|---|---|---|---|---|
| D/rf_shallow | +0.235 | [-0.039, +0.428] | 95.4% | cross-population |
| B/rf_shallow | +0.176 | [-0.021, +0.352] | 91.2% | cross-population |
| D/lr_l2_baseline | +0.192 | [-0.059, +0.410] | 87.1% | cross-population |
| **informed_prior_flipped, lam=5** | **+0.068** | **[-0.010, +0.149]** | **92.3%** | **clean - same population as baseline** |
| D/gaussian_nb | +0.086 | [-0.187, +0.259] | 77.3% | cross-population |
| B/gaussian_nb | +0.075 | [-0.174, +0.241] | 77.3% | cross-population |

(200 resamples, reduced from an initial 500 after RandomForest refits repeatedly exceeded available runtime even from cached feature data - same day-block-resample procedure, smaller N. 194 of 200 resamples were valid for every candidate, all evaluated on the identical shared subsequence.)

**What actually changes vs. the marginal-CI conclusion**: every candidate now shows a strong directional signal - 77-95% of individual paired resamples favor the candidate over baseline, not the coin-flip you'd expect from pure noise. The marginal comparison's overlap was hiding this because shared day-to-day noise was swamping the signal in both directions equally. That part of your instinct was right: marginal CIs understated how consistent the advantage is.

**But strict 95% significance still isn't cleared by any candidate** - every CI still technically contains zero. The closest by far is **`informed_prior_flipped` at lam=5**: CI [-0.010, +0.149], barely touching zero at the low end, 92.3% win rate, and - critically - it's the *only* comparison here with no cross-population caveat attached. Of everything tested across `informed_prior/`, `diagnostics/`, and this entire sweep, this is the single most credible candidate result produced so far: not a confirmed win, but the first one that's both nearly significant *and* methodologically clean.

The D/B RandomForest and D/lr_l2_baseline rows have larger point diffs and higher win rates, but I'd trust them less: they carry the population-mismatch caveat on top of not clearing significance either, so their more dramatic numbers are the least reliable of the six, not the most promising.

**Revised verdict**: still no confirmed winner at strict 95% significance - but this is no longer "everything washes out to indistinguishable noise." The paired test surfaces a real, consistent directional signal, and `informed_prior_flipped` (lam=5) is close enough, and clean enough, to be worth specifically re-testing once more personal data accumulates, rather than filed away with everything else that landed at baseline.

## Files

- `feature_variants.py` - Part 1 (feature sets A/B/C/D)
- `models.py` - Part 2 model implementations (nested ElasticNet, conditional logit; simple classifiers reuse `day_based_cv` directly)
- `sample_weighting.py` - Part 3 (dirty-positive tagging, all-negatives collection)
- `augmentation.py` - Part 4 (jitter, magnitude warp, time warp; train-fold-only augmented CV)
- `run_sweep.py` - orchestrates Parts 1-4, logs every result to `results_log.jsonl`
- `bootstrap_top5.py`, `bootstrap_informed_prior_flipped.py` - standalone bootstrap continuations (split out after `run_sweep.py`'s single-process bootstrap phase hit external time limits twice)
- `paired_bootstrap.py` - paired day-block bootstrap follow-up, with per-feature-set dataset caching (`_dataset_cache_*.npz`) and per-candidate chunk checkpointing (RandomForest refits needed both to fit within available runtime)
- `results_log.jsonl` - all 46 raw CV results
- `bootstrap_results.json`, `bootstrap_informed_prior_flipped.json` - marginal bootstrap outputs
- `paired_bootstrap_results.json` - paired bootstrap outputs backing the table above, including full per-resample arrays
