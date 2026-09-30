# Experiment B: 50-Replication Diagnostic Report

## Design

- Scenario: standard-Heston regimes with matched stationary CIR variance laws
  and different mean-reversion speeds.
- For every path, all regimes share `theta`, `mu`, `rho`, and
  `sigma^2 / kappa`; only `kappa` has regime-specific ranges.
- Independent replications: 50.
- Paths per replication: 10 train, 3 validation, 5 test.
- Observations per path: 5,000; causal window: 50 observations.
- Representations: Statistics (11 features), raw Signature level 3
  (39 features), and Combined (50 features).
- Fixed classifier: Random Forest, 200 trees, maximum depth 6, balanced class
  weights.
- Simulation uses four Euler substeps per observed return and stationary CIR
  initialization. The discretization audit in `results/discretization_audit`
  finds that the stationary variance of V differs across the three `kappa`
  values by about 1 percentage point with four substeps, versus about 3 points
  with one Euler step.
- Pairing: every representation uses the exact same paths and labels in each
  replication.
- Detector settings are selected on validation paths only from a fixed grid of
  EMA alpha, probability threshold, and consecutive confirmations. Test paths
  are evaluated once. A predicted transition is matched only if it occurs
  after a true change, enters the correct new state, and falls within 100
  observations.

These frozen results use raw signatures. The earlier logsignature preparation
failure came from the PyPI `iisignature 0.24` wheel on macOS arm64, not from
NumPy 2.0 itself. The workspace now uses a pinned source build that supports
logsignatures; the revised 10/10/5 main grid defaults to them but has not been
run, so the numbers below remain the original raw-signature results.

## Pointwise state classification

| Metric | Mean | 95% t confidence interval |
|---|---:|---:|
| Balanced accuracy, Statistics | 0.40071 | [0.39542, 0.40599] |
| Balanced accuracy, Signature | 0.39701 | [0.39105, 0.40297] |
| Balanced accuracy, Combined | **0.40675** | **[0.40106, 0.41244]** |
| Combined minus Statistics | **+0.00604** | **[+0.00295, +0.00913]** |

Combined improves balanced accuracy by about 0.60 percentage points over
Statistics. The paired interval excludes zero, but the absolute performance is
low and the effect is small. Signature alone does not outperform Statistics.

Mean per-regime recall shows a representation tradeoff:

| Representation | Slow `kappa` (0) | Medium `kappa` (1) | Fast `kappa` (2) | Macro F1 |
|---|---:|---:|---:|---:|
| Statistics | 0.390 | **0.283** | 0.530 | 0.386 |
| Signature | 0.304 | 0.226 | **0.661** | 0.368 |
| Combined | 0.352 | 0.265 | 0.603 | **0.387** |

The classifier is biased toward the fast mean-reversion regime. The expanded
classwise diagnostics show that Signature's higher fast recall is driven mainly
by predicting fast more often, not by more precise fast-regime recognition:

| Fast-regime metric | Statistics | Signature | Combined | Signature - Statistics [95% CI] |
|---|---:|---:|---:|---:|
| True prevalence | 0.329 | 0.329 | 0.329 | 0 |
| Predicted proportion | 0.426 | **0.567** | 0.490 | **+0.141 [+0.118, +0.164]** |
| Recall | 0.530 | **0.661** | 0.603 | **+0.131 [+0.106, +0.156]** |
| Precision | **0.409** | 0.383 | 0.405 | **-0.025 [-0.032, -0.019]** |
| F1 | 0.455 | **0.479** | 0.478 | +0.023 [+0.013, +0.033] |
| One-vs-rest ROC-AUC | **0.617** | 0.603 | 0.627 | **-0.014 [-0.022, -0.007]** |
| One-vs-rest PR-AUC | **0.430** | 0.400 | 0.434 | **-0.030 [-0.039, -0.020]** |
| One-vs-rest Brier | **0.2125** | 0.2144 | 0.2114 | **+0.0019 [+0.0010, +0.0029]** |

Signature predicts fast in 56.7% of windows although only 32.9% are truly
fast. Its predicted fast proportion exceeds Statistics by 14.1 percentage
points in 98% of replications. The recall gain is therefore a threshold/frequency
effect accompanied by worse precision, ranking discrimination, and Brier score.
It should be interpreted as fast-class overprediction, not improved precision.

This classwise decomposition was added after inspecting the primary test
results. It is therefore an exploratory diagnosis on the same test paths, not
a pre-registered confirmatory claim. Its intervals quantify Monte Carlo
variation in this sample but do not correct for post-hoc metric selection.

All slow, medium, and fast classwise precision, recall, F1, prevalence,
predicted proportion, ROC-AUC, PR-AUC, and Brier results are recorded in
`classwise_metrics.csv` and `classwise_summary.csv`. Paired representation
contrasts are in `classwise_contrasts.csv`; pooled 10-bin reliability data are
in `calibration_bins.csv`.

Aggregate row-normalized confusion matrices (rows are true states, columns are
predicted states):

### Statistics

| True / Predicted | 0 | 1 | 2 |
|---|---:|---:|---:|
| 0 | 0.385 | 0.280 | 0.335 |
| 1 | 0.304 | 0.283 | 0.413 |
| 2 | 0.216 | 0.253 | 0.531 |

### Signature

| True / Predicted | 0 | 1 | 2 |
|---|---:|---:|---:|
| 0 | 0.303 | 0.212 | 0.485 |
| 1 | 0.216 | 0.228 | 0.556 |
| 2 | 0.147 | 0.193 | 0.660 |

### Combined

| True / Predicted | 0 | 1 | 2 |
|---|---:|---:|---:|
| 0 | 0.350 | 0.258 | 0.392 |
| 1 | 0.257 | 0.267 | 0.476 |
| 2 | 0.169 | 0.229 | 0.603 |

## Causal transition detection

| Metric | Statistics | Signature | Combined |
|---|---:|---:|---:|
| Detector balanced accuracy | 0.418 | 0.387 | 0.421 |
| Boundary F1 | 0.113 | 0.101 | 0.114 |
| Detection probability within 100 steps | 0.235 | 0.204 | 0.252 |
| False switches per 1,000 windows | 5.10 | 4.57 | 5.28 |
| Mean matched-change delay | 50.67 | 48.15 | 51.79 |

None of the paired Combined-minus-Statistics intervals for detector balanced
accuracy, boundary F1, detection probability, false switches, or delay excludes
zero. In particular, the mean boundary-F1 difference is +0.0013 with a 95%
interval of [-0.0132, +0.0158]. The experiment therefore supports a small
pointwise recognition gain, not a transition-detection gain.

The event analysis is exploratory. Each replication contains about 46 true
test changes on average, but only three validation paths tune 27 detector
settings. Validation boundary F1 has negative correlation with test boundary
F1 for all three representations. A stronger event-level study should increase
the number of validation/test paths, reduce or nest the tuning grid, and report
delay-versus-false-alarm curves rather than one selected operating point.

Mean delay excludes replications with no correctly matched test transition:
two Statistics runs, two Signature runs, and one Combined run. Detection
probability and boundary F1 retain those failures as zero.

## Interpretation relative to Experiment A

Experiment A is now a pure scale-only design; Combined balanced accuracy is
0.74664. Experiment B removes variance-scale separation and drops Combined
balanced accuracy to 0.40675. The Combined-minus-Statistics gains are small in
both experiments: 0.74 percentage points in A and 0.60 percentage points in B.
This is not evidence that signatures solve dynamic-regime detection. Raw
Signature alone is weaker than Statistics, and no online event metric improves
reliably.

## Output files

- `replications.csv`: one wide row per replication, including class counts,
  state diagnostics, selected detector settings, and event metrics.
- `class_counts.csv`: tidy split-level counts for regimes 0/1/2.
- `test_diagnostics.csv`: tidy per-replication/per-representation state,
  calibration, and event metrics.
- `classwise_metrics.csv`: one row per replication, representation, and class
  with precision, recall, F1, prevalence, predicted proportion, ROC-AUC,
  PR-AUC, and one-vs-rest Brier score.
- `classwise_summary.csv`: replication-level means and 95% intervals for all
  classwise metrics.
- `classwise_contrasts.csv`: paired Signature/Combined-minus-Statistics
  classwise contrasts and win rates.
- `calibration_bins.csv`: pooled fixed-bin reliability-curve data for every
  representation and class.
- `summary.csv`: means and Student-t 95% confidence intervals across independent
  replications, including paired differences.
- `config.json`: frozen Experiment B and detector-grid configuration.
