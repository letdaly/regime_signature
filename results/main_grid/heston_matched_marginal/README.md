# Main grid, row 2: standard Heston, matched-marginal regimes

## Configuration

Main-grid configuration (rerun 2026-09-19): 50 paired replications, 10 train / 10 validation / 5 test paths,
5,000 observations per path, 50-step causal windows, **level-3 log-signatures** (14 coordinates) on the
origin-anchored three-channel path (`signature_path = origin_anchored_w_increments`; the 2026-09-18 run,
archived as `heston_matched_marginal_path49/`, dropped the first return of each window from the signature), Statistics 11
features, Combined 25. Two learners on the same column subsets: the fixed Random Forest (200 trees, depth 6,
balanced; bare column names) and a multinomial logistic regression (robust scaling and tail clipping fitted on
training windows, C in {0.01, 0.1, 1} selected by validation balanced accuracy; `logistic_` columns).
Causal detector tuned on validation paths only (27-point grid) for every learner-representation cell, event
tolerance 100. Base seed unchanged, so replication k draws the same parameter families as the archived run.

Design as in `results/experiment_b_diagnostics`: within each path all regimes share theta, mu, rho, and sigma^2/kappa; only kappa changes, so the stationary CIR law of the variance is identical across regimes and only its time scale differs. Four Euler substeps per observation. Classwise probability diagnostics and calibration bins are saved for both learners.

## Random Forest (main-table learner)

### Pointwise classification

| Metric | Mean | 95% t CI | 2026-09-18 run (49-increment path, forest) |
|---|---:|---:|---:|
| Balanced accuracy, Statistics | 0.3952 | [0.3896, 0.4007] | 0.3952 |
| Balanced accuracy, Signature | 0.3998 | [0.3938, 0.4058] | 0.3990 |
| Balanced accuracy, Combined | 0.4027 | [0.3972, 0.4081] | 0.4018 |
| Combined minus Statistics (43 of 50 positive) | **+0.0075** | **[+0.0053, +0.0097]** | +0.0066 |
| Signature minus Statistics (30 of 50 positive) | +0.0047 | [-0.0005, +0.0098] | (from replications.csv) |

Per-regime test recall (mean over replications):

| Representation | Regime 0 | Regime 1 | Regime 2 | Macro F1 |
|---|---:|---:|---:|---:|
| Statistics | 0.386 | 0.281 | 0.519 | 0.380 |
| Signature | 0.449 | 0.245 | 0.505 | 0.379 |
| Combined | 0.395 | 0.270 | 0.543 | 0.385 |

### Causal detector on test paths

| Metric | Statistics | Signature | Combined | Combined - Statistics [95% CI] |
|---|---:|---:|---:|---:|
| Detector balanced accuracy | 0.4151 | 0.4312 | 0.4224 | +0.0073 [-0.0025, +0.0171] |
| Boundary F1 | 0.114 | 0.120 | 0.115 | +0.001 [-0.008, +0.011] |
| Detection probability within 100 | 0.227 | 0.201 | 0.219 | -0.008 [-0.043, +0.027] |
| Mean matched delay (obs.) | 49.1 | 50.5 | 49.6 | +0.454 [-2.272, +3.181] |
| False switches per 1,000 windows | 4.93 | 3.74 | 4.79 | -0.146 [-1.001, +0.709] |

### Balanced accuracy by observations since the last true regime change

| Bucket | Windows per replication | Statistics | Signature | Combined | Combined - Statistics [95% CI] |
|---|---:|---:|---:|---:|---:|
| 0-24 | 1150 | 0.3177 | 0.3136 | 0.3146 | -0.0031 [-0.0086, +0.0024] |
| 25-49 | 1142 | 0.3778 | 0.3883 | 0.3853 | **+0.0075 [+0.0009, +0.0140]** |
| 50-99 | 2150 | 0.4014 | 0.4131 | 0.4145 | **+0.0131 [+0.0083, +0.0178]** |
| 100+ | 20313 | 0.4001 | 0.4039 | 0.4074 | **+0.0074 [+0.0050, +0.0097]** |

## Logistic regression (second learner)

Selected regularization: statistics: C=0.01 x15, C=0.1 x9, C=1 x26; signature: C=0.01 x18, C=0.1 x11, C=1 x21; combined: C=0.01 x12, C=0.1 x17, C=1 x21.

### Pointwise classification

| Metric | Mean | 95% t CI |
|---|---:|---:|
| Balanced accuracy, Statistics | 0.3978 | [0.3911, 0.4044] |
| Balanced accuracy, Signature | 0.3953 | [0.3895, 0.4010] |
| Balanced accuracy, Combined | 0.4166 | [0.4106, 0.4225] |
| Combined minus Statistics (49 of 50 positive) | **+0.0188** | **[+0.0159, +0.0217]** |
| Signature minus Statistics (19 of 50 positive) | -0.0025 | [-0.0091, +0.0041] |

Per-regime test recall (mean over replications):

| Representation | Regime 0 | Regime 1 | Regime 2 | Macro F1 |
|---|---:|---:|---:|---:|
| Statistics | 0.472 | 0.237 | 0.484 | 0.382 |
| Signature | 0.547 | 0.235 | 0.404 | 0.380 |
| Combined | 0.505 | 0.260 | 0.485 | 0.402 |

### Causal detector on test paths

| Metric | Statistics | Signature | Combined | Combined - Statistics [95% CI] |
|---|---:|---:|---:|---:|
| Detector balanced accuracy | 0.4072 | 0.4262 | 0.4387 | **+0.0314 [+0.0187, +0.0442]** |
| Boundary F1 | 0.120 | 0.113 | 0.121 | +0.000 [-0.012, +0.013] |
| Detection probability within 100 | 0.311 | 0.252 | 0.231 | **-0.080 [-0.141, -0.019]** |
| Mean matched delay (obs.) | 48.3 | 51.8 | 49.5 | +1.174 [-1.675, +4.023] |
| False switches per 1,000 windows | 7.04 | 5.46 | 4.63 | **-2.409 [-3.916, -0.903]** |

### Balanced accuracy by observations since the last true regime change

| Bucket | Windows per replication | Statistics | Signature | Combined | Combined - Statistics [95% CI] |
|---|---:|---:|---:|---:|---:|
| 0-24 | 1150 | 0.3220 | 0.3142 | 0.3089 | **-0.0132 [-0.0247, -0.0017]** |
| 25-49 | 1142 | 0.3845 | 0.3772 | 0.3972 | **+0.0128 [+0.0041, +0.0214]** |
| 50-99 | 2150 | 0.4020 | 0.4032 | 0.4289 | **+0.0268 [+0.0184, +0.0352]** |
| 100+ | 20313 | 0.4028 | 0.4002 | 0.4228 | **+0.0200 [+0.0165, +0.0235]** |

## Learner contrast (paired, same windows)

| Representation | Logistic minus Random Forest balanced accuracy [95% CI] |
|---|---:|
| Statistics | +0.0026 [-0.0024, +0.0076] |
| Signature | -0.0046 [-0.0096, +0.0005] |
| Combined | **+0.0139 [+0.0095, +0.0183]** |

Fast-regime (regime 2) prediction share versus its prevalence:

| Learner | Representation | Predicted fast share | Prevalence |
|---|---|---:|---:|
| random_forest | Statistics | 0.417 | 0.301 |
| random_forest | Signature | 0.402 | 0.301 |
| random_forest | Combined | 0.431 | 0.301 |
| logistic | Statistics | 0.384 | 0.301 |
| logistic | Signature | 0.319 | 0.301 |
| logistic | Combined | 0.364 | 0.301 |

## Reading

- **Forest: same picture as the archived run, slightly larger.** Combined minus Statistics **+0.0075 [+0.0053, +0.0097]** (43 of
  50 positive; archived +0.0066 [+0.0045, +0.0087]); Signature alone on par with Statistics (+0.0047 mean, CI includes zero). Every forest detector
  contrast includes zero (detector balanced accuracy +0.0073 [-0.0025, +0.0171], boundary F1 +0.001 [-0.008, +0.011], detection probability
  -0.008 [-0.043, +0.027], delay +0.45 [-2.27, +3.18], false switches -0.15 [-1.00, +0.71]). The gain sits in windows lying entirely inside the new
  regime (50-99: +0.0131 [+0.0083, +0.0178]; 100+: +0.0074 [+0.0050, +0.0097]) and now also, marginally, in 25-49 (+0.0075 [+0.0009, +0.0140]); nothing in 0-24.
- **The logistic learner extracts a much larger and more consistent signature increment**: **+0.0188 [+0.0159, +0.0217]**, positive
  in 49 of 50 replications, and this time it is not baseline weakness: logistic Statistics matches forest
  Statistics (+0.0026 [-0.0024, +0.0076]) while logistic Combined beats forest Combined by **+0.0139 [+0.0095, +0.0183]**. Logistic
  Combined is the most accurate cell of the row (0.4166). Signature alone does not help the linear model
  (-0.0025 mean, 19 of 50 positive); the increment needs both blocks, consistent with the mechanism benchmark's B3 + signature result
  at 10/5/5 (`results/beyond_marginal_benchmark`, +0.0306 for logistic, +0.0001 for the forest).
- **The logistic gain is spread across all windows inside the new regime** (25-49: +0.0128 [+0.0041, +0.0214]; 50-99: +0.0268 [+0.0184, +0.0352];
  100+: +0.0200 [+0.0165, +0.0235]) and is *negative* in the first 25 observations after a switch (-0.0132 [-0.0247, -0.0017]): the linear model
  with signatures commits to the new state later than the moment-only linear model.
- **Event-level: operating-point dependent, not a clean improvement.** For the logistic learner, adding
  signatures raises detector balanced accuracy (+0.0314 [+0.0187, +0.0442]) and cuts false switches (-2.41 [-3.92, -0.90]), but lowers
  detection probability (-0.080 [-0.141, -0.019]); boundary F1 (+0.000 [-0.012, +0.013]) and delay (+1.17 [-1.68, +4.02]) include zero. The logistic
  Statistics detector is trigger-happy (7.04 false switches per 1,000 vs 4.93 for the forest,
  detection probability 0.311), and the signature makes it more conservative. Since the selected
  operating points do not impose a common false-alarm budget, this is a trade-off rather than a dominance
  result.
- **Fast-regime bias.** The forest still over-predicts the fast regime (predicted share 0.431 for
  Combined against a prevalence of 0.301); the logistic learner's share is closer to prevalence (0.364 for
  Combined, 0.319 for Signature alone) and its Combined gain is mostly slow-regime recall (recall 0/1/2 =
  0.505/0.260/0.485 versus 0.472/0.237/0.484 for logistic Statistics).

## Output files

`config.json`, `replications.csv`, `summary.csv`, `class_counts.csv`, `test_diagnostics.csv`,
`classwise_metrics.csv`, `classwise_summary.csv`, `classwise_contrasts.csv`, `calibration_bins.csv`; every tidy
table carries a `learner` column.
