# Experiment A: Scale-Only 50-Replication Study with Causal Transition Detector

## Relation to `results/experiment_a_scale_only`

Same design, seeds, paths, features, and Random Forests as the frozen
classification-only study (`base_seed` 20260827, 10/3/5 paths, 5,000
observations, 50-step windows, 200 trees, `signature_kind="raw"`, level 3).
All 68 classification columns of `replications.csv` are bit-identical to the
frozen study across all 50 replications; this directory adds the event-level
metrics required for every cell of the main table.

## Detector protocol

- Random Forest probabilities (fixed regime order) are smoothed causally by an
  EMA with weight `alpha`, a state change is proposed when the smoothed argmax
  differs from the current state and its probability is at least `threshold`,
  and it is accepted after `confirmations` consecutive proposals.
- The 27-point grid (`alpha` in {0.2, 0.5, 1.0}, `threshold` in {0.4, 0.5,
  0.6}, `confirmations` in {1, 3, 5}) is searched **on validation paths
  only**, ranked by boundary F1, then detection probability, fewer false
  switches, and shorter delay. Test paths are evaluated once.
- A predicted transition matches a true change if it occurs after the change,
  enters the correct new state, and falls within 100 observations
  (`event_tolerance`). Unmatched predicted transitions are false switches.
- Test paths contain about 46 true changes per replication.
- Implementation: `regime_detection.py`, shared with Experiment B.

## Pointwise classification (unchanged from the frozen study)

| Metric | Mean | 95% t confidence interval |
|---|---:|---:|
| Balanced accuracy, Statistics | 0.7392 | [0.7323, 0.7461] |
| Balanced accuracy, Signature | 0.7442 | [0.7373, 0.7512] |
| Balanced accuracy, Combined | 0.7466 | [0.7395, 0.7537] |
| Combined minus Statistics | **+0.0074** | **[+0.0060, +0.0088]** |

## Causal detector on test paths

| Metric | Statistics | Signature | Combined |
|---|---:|---:|---:|
| Detector balanced accuracy | 0.7334 | 0.7397 | 0.7423 |
| Boundary F1 | 0.258 | 0.257 | 0.264 |
| Detection probability within 100 | 0.464 | 0.492 | 0.497 |
| Mean matched delay (observations) | 65.5 | 64.3 | 64.4 |
| False switches per 1,000 windows | 3.99 | 4.34 | 4.25 |

Paired contrasts, Combined minus Statistics (50 replications, Student-t):

| Metric | Mean difference | 95% t confidence interval | Combined better |
|---|---:|---:|---:|
| Detector balanced accuracy | **+0.0089** | **[+0.0073, +0.0105]** | 47 of 50 |
| Boundary F1 | +0.0060 | [-0.0034, +0.0154] | 33 of 50 |
| Detection probability | **+0.0325** | **[+0.0161, +0.0490]** | 32 of 50 |
| Mean delay (negative is faster) | -1.10 | [-2.32, +0.12] | 30 of 50 |
| False switches per 1,000 (negative is fewer) | +0.258 | [+0.011, +0.505] | 18 of 50 |

## Reading

- The pointwise gain survives the causal smoothing stage: detector balanced
  accuracy rises by about 0.9 percentage points with Combined features, and
  the interval excludes zero (positive in 47 of 50 replications).
- At the event level the picture is the same as in Experiment B, only with
  higher absolute numbers because the regimes are well separated: Combined
  detects about 3 percentage points more true changes within 100 observations
  (interval excludes zero), but pays for it with roughly 0.26 more false
  switches per 1,000 windows (interval excludes zero on the wrong side). The
  net boundary F1 improvement (+0.006) and the delay reduction (about one
  observation) do not exclude zero.
- Mean delays of about 65 observations exceed the 50-step window: the
  classifier needs roughly one full fresh-regime window before its smoothed
  probabilities cross the threshold. Detection probability below 0.5 means
  that half of the true changes are not confirmed within 100 observations at
  the validation-selected operating point.
- Selected operating points cluster at `threshold = 0.6` (`alpha` 0.2 with 1
  or 5 confirmations most often), so the validation stage prefers conservative
  detectors; see `test_diagnostics.csv` for the per-replication choice. With
  only 3 validation paths this selection is noisy; the rerun with 10
  validation paths is `results/main_grid/heston_scale_only`.

## Output files

- `config.json`: frozen configuration including `detector_grid` and `event_tolerance`
- `replications.csv`: one row per paired replication (classification + detector columns)
- `summary.csv`: replication-level means and Student-t intervals for all metrics
- `class_counts.csv`: class counts by split and replication
- `test_diagnostics.csv`: per-representation recalls, confusion matrices, detector metrics, and selected operating points
