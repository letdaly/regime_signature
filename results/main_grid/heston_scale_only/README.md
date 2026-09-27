# Main grid, row 1: standard Heston, scale-only regimes

## Configuration

Main-grid configuration (rerun 2026-09-19): 50 paired replications, 10 train / 10 validation / 5 test paths,
5,000 observations per path, 50-step causal windows, **level-3 log-signatures** (14 coordinates) on the
origin-anchored three-channel path (`signature_path = origin_anchored_w_increments`; the 2026-09-18 run,
archived as `heston_scale_only_path49/`, dropped the first return of each window from the signature), Statistics 11
features, Combined 25. Two learners on the same column subsets: the fixed Random Forest (200 trees, depth 6,
balanced; bare column names) and a multinomial logistic regression (robust scaling and tail clipping fitted on
training windows, C in {0.01, 0.1, 1} selected by validation balanced accuracy; `logistic_` columns).
Causal detector tuned on validation paths only (27-point grid) for every learner-representation cell, event
tolerance 100. Base seed unchanged, so replication k draws the same parameter families as the archived run.

Design as in `results/experiment_a_scale_only`: within each path all regimes share mu, kappa, rho, and c; only theta_i changes, sigma_i = c sqrt(kappa theta_i).

## Random Forest (main-table learner)

### Pointwise classification

| Metric | Mean | 95% t CI | 2026-09-18 run (49-increment path, forest) |
|---|---:|---:|---:|
| Balanced accuracy, Statistics | 0.7353 | [0.7275, 0.7431] | 0.7353 |
| Balanced accuracy, Signature | 0.7306 | [0.7229, 0.7384] | 0.7300 |
| Balanced accuracy, Combined | 0.7382 | [0.7305, 0.7460] | 0.7380 |
| Combined minus Statistics (44 of 50 positive) | **+0.0029** | **[+0.0022, +0.0036]** | +0.0027 |
| Signature minus Statistics (13 of 50 positive) | -0.0047 | [-0.0068, -0.0026] | -0.0054 |

Per-regime test recall (mean over replications):

| Representation | Regime 0 | Regime 1 | Regime 2 | Macro F1 |
|---|---:|---:|---:|---:|
| Statistics | 0.781 | 0.702 | 0.723 | 0.734 |
| Signature | 0.771 | 0.703 | 0.718 | 0.730 |
| Combined | 0.782 | 0.708 | 0.725 | 0.737 |

### Causal detector on test paths

| Metric | Statistics | Signature | Combined | Combined - Statistics [95% CI] |
|---|---:|---:|---:|---:|
| Detector balanced accuracy | 0.7338 | 0.7030 | 0.7360 | **+0.0021 [+0.0005, +0.0038]** |
| Boundary F1 | 0.270 | 0.256 | 0.270 | -0.000 [-0.005, +0.005] |
| Detection probability within 100 | 0.488 | 0.426 | 0.481 | -0.007 [-0.018, +0.005] |
| Mean matched delay (obs.) | 64.4 | 64.5 | 63.7 | -0.651 [-1.496, +0.194] |
| False switches per 1,000 windows | 4.01 | 3.54 | 3.92 | -0.091 [-0.233, +0.050] |

### Balanced accuracy by observations since the last true regime change

| Bucket | Windows per replication | Statistics | Signature | Combined | Combined - Statistics [95% CI] |
|---|---:|---:|---:|---:|---:|
| 0-24 | 1155 | 0.1106 | 0.1164 | 0.1098 | -0.0008 [-0.0026, +0.0011] |
| 25-49 | 1150 | 0.1896 | 0.1943 | 0.1908 | +0.0013 [-0.0015, +0.0040] |
| 50-99 | 2143 | 0.4467 | 0.4468 | 0.4486 | +0.0019 [-0.0007, +0.0045] |
| 100+ | 20307 | 0.8346 | 0.8284 | 0.8381 | **+0.0034 [+0.0027, +0.0042]** |

## Logistic regression (second learner)

Selected regularization: statistics: C=0.01 x10, C=0.1 x26, C=1 x14; signature: C=0.01 x10, C=0.1 x13, C=1 x27; combined: C=0.01 x12, C=0.1 x18, C=1 x20.

### Pointwise classification

| Metric | Mean | 95% t CI |
|---|---:|---:|
| Balanced accuracy, Statistics | 0.7303 | [0.7227, 0.7378] |
| Balanced accuracy, Signature | 0.7355 | [0.7286, 0.7423] |
| Balanced accuracy, Combined | 0.7385 | [0.7312, 0.7458] |
| Combined minus Statistics (46 of 50 positive) | **+0.0083** | **[+0.0068, +0.0097]** |
| Signature minus Statistics (37 of 50 positive) | +0.0052 | [+0.0030, +0.0074] |

Per-regime test recall (mean over replications):

| Representation | Regime 0 | Regime 1 | Regime 2 | Macro F1 |
|---|---:|---:|---:|---:|
| Statistics | 0.803 | 0.692 | 0.696 | 0.729 |
| Signature | 0.834 | 0.643 | 0.729 | 0.732 |
| Combined | 0.810 | 0.701 | 0.705 | 0.737 |

### Causal detector on test paths

| Metric | Statistics | Signature | Combined | Combined - Statistics [95% CI] |
|---|---:|---:|---:|---:|
| Detector balanced accuracy | 0.7255 | 0.7129 | 0.7333 | **+0.0078 [+0.0044, +0.0113]** |
| Boundary F1 | 0.258 | 0.282 | 0.275 | **+0.017 [+0.009, +0.026]** |
| Detection probability within 100 | 0.477 | 0.486 | 0.514 | **+0.037 [+0.012, +0.063]** |
| Mean matched delay (obs.) | 63.3 | 62.1 | 61.9 | **-1.454 [-2.727, -0.181]** |
| False switches per 1,000 windows | 4.17 | 3.70 | 4.21 | +0.040 [-0.263, +0.343] |

### Balanced accuracy by observations since the last true regime change

| Bucket | Windows per replication | Statistics | Signature | Combined | Combined - Statistics [95% CI] |
|---|---:|---:|---:|---:|---:|
| 0-24 | 1155 | 0.1219 | 0.1240 | 0.1200 | -0.0020 [-0.0065, +0.0025] |
| 25-49 | 1150 | 0.2102 | 0.2534 | 0.2388 | **+0.0286 [+0.0232, +0.0340]** |
| 50-99 | 2143 | 0.4572 | 0.5065 | 0.4858 | **+0.0286 [+0.0228, +0.0343]** |
| 100+ | 20307 | 0.8254 | 0.8238 | 0.8309 | **+0.0055 [+0.0038, +0.0073]** |

## Learner contrast (paired, same windows)

| Representation | Logistic minus Random Forest balanced accuracy [95% CI] |
|---|---:|
| Statistics | **-0.0050 [-0.0075, -0.0026]** |
| Signature | **+0.0048 [+0.0020, +0.0077]** |
| Combined | +0.0003 [-0.0024, +0.0030] |

## Reading

- **Path fix changes nothing material for the forest.** Combined minus Statistics is **+0.0029 [+0.0022, +0.0036]** (44 of 50
  positive) against +0.0027 [+0.0021, +0.0033] in the archived run; Signature alone remains below Statistics (-0.0047 [-0.0068, -0.0026]).
  The Statistics columns are identical to the archived run by construction.
- **The one detector contrast that excluded zero in the archived run no longer does.** Mean matched delay is
  -0.65 [-1.50, +0.19] (archived: -1.15 [-2.00, -0.30]); detector balanced accuracy +0.0021 [+0.0005, +0.0038] still excludes zero, boundary F1,
  detection probability and false switches do not. The one-observation delay reduction reported on 2026-09-18
  should be treated as fragile.
- **The forest gain still lives in mature windows** (100+: +0.0034 [+0.0027, +0.0042]; the three transition buckets include zero).
- **The logistic learner reaches the same Combined accuracy as the forest** (+0.0003 [-0.0024, +0.0030]) from a weaker
  Statistics baseline (-0.0050 [-0.0075, -0.0026]), so its larger within-learner increment (**+0.0083 [+0.0068, +0.0097]**, 46 of 50) is
  partly baseline weakness and partly a better use of the signature coordinates (+0.0048 [+0.0020, +0.0077] for Signature
  alone, where the linear model beats the forest). For this learner Signature alone also beats Statistics
  (+0.0052 [+0.0030, +0.0074]).
- **With the logistic learner the signature increment is visible in event terms.** Boundary F1 +0.017 [+0.009, +0.026],
  detection probability +0.037 [+0.012, +0.063] and detector balanced accuracy +0.0078 [+0.0044, +0.0113] exclude zero, delay is 1.45 observations
  shorter (-1.45 [-2.73, -0.18]), and false switches do not change (+0.04 [-0.26, +0.34]). Stratified by time since the last change, the logistic gain is
  +0.0286 [+0.0232, +0.0340] at 25-49 and +0.0286 [+0.0228, +0.0343] at 50-99 observations, i.e. in windows that straddle or just follow the switch,
  and +0.0055 [+0.0038, +0.0073] in mature windows; nothing at 0-24. The logistic Statistics detector is the weaker starting point
  (boundary F1 0.258 vs 0.270 for the forest), so these are within-learner contrasts, not evidence
  that the logistic detector is the better one in absolute terms.

## Output files

`config.json`, `replications.csv`, `summary.csv`, `class_counts.csv`, `test_diagnostics.csv` (per learner and
representation: recalls, confusion matrices, detector metrics and operating points, stratified BA, selected C).
