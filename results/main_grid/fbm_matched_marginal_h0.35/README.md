# Main grid, row 3: fBM-driven stochastic volatility, matched-marginal regimes, H = 0.35

## Configuration

Run 2026-09-20: 50 paired replications, 10 train / 10 validation / 5 test paths, 5,000 observations per path,
50-step causal windows, level-3 log-signatures (14 coordinates) on the origin-anchored three-channel path
(`signature_path = origin_anchored_w_increments`), Statistics 11 features, Combined 25. Two learners on the same
column subsets: the fixed Random Forest (200 trees, depth 6, balanced; bare column names) and a multinomial
logistic regression (robust scaling and tail clipping fitted on training windows, C in {0.01, 0.1, 1} selected
by validation balanced accuracy; `logistic_` columns). Causal detector tuned on validation paths only (27-point
grid) for every learner-representation cell, event tolerance 100.

Simulator: `FBMStochasticVolatilitySimulator` (exact Hosking fGn with Hurst index 0.35 replacing the Brownian
increments of the CIR variance equation; the price shock uses the fGn *innovation*, so log returns stay serially
uncorrelated). Parameter families, base seed and replication seeds are identical to the paired Heston row
`results/main_grid/heston_matched_marginal` (`sigma_scale = 1`, 4 substeps per observation), so
each replication here is the same family under rough variance noise. The Gamma stationary initialisation is exact
only at H = 0.5; the audit below quantifies the resulting marginal mismatch (identical families, mismatch stated as a limitation).

## Stationary variance audit (first replication's family, 10,000 no-switch observations)

| Setting | Regime | kappa | sigma | mean(V) | sd(V) | CIR sd | sd ratio to Heston | Return excess kurtosis | acf1(dV) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rough | 0 | 0.83 | 0.210 | 0.0711 | 0.0143 | 0.0460 | 0.27 | 0.21 | -0.20 |
| rough | 1 | 4.61 | 0.495 | 0.0685 | 0.0189 | 0.0460 | 0.42 | 0.31 | -0.19 |
| rough | 2 | 14.88 | 0.890 | 0.0698 | 0.0214 | 0.0460 | 0.51 | 0.20 | -0.20 |
| heston_reference | 0 | 0.83 | 0.210 | 0.0785 | 0.0526 | 0.0460 | - | 1.54 | +0.01 |
| heston_reference | 1 | 4.61 | 0.495 | 0.0784 | 0.0450 | 0.0460 | - | 0.86 | -0.00 |
| heston_reference | 2 | 14.88 | 0.890 | 0.0773 | 0.0419 | 0.0460 | - | 0.76 | -0.03 |

## Random Forest (main-table learner)

### Pointwise classification

| Metric | Mean | 95% t CI | Paired Heston row |
|---|---:|---:|---:|
| Balanced accuracy, Statistics | 0.3583 | [0.3530, 0.3635] | 0.3952 |
| Balanced accuracy, Signature | 0.3582 | [0.3544, 0.3621] | 0.3998 |
| Balanced accuracy, Combined | 0.3613 | [0.3561, 0.3664] | 0.4027 |
| Combined minus Statistics (33 of 50 positive) | **+0.0030** | **[+0.0014, +0.0046]** | +0.0075 |
| Signature minus Statistics (24 of 50 positive) | -0.0000 | [-0.0052, +0.0052] | +0.0047 |

Per-regime test recall (mean over replications):

| Representation | Regime 0 | Regime 1 | Regime 2 | Macro F1 |
|---|---:|---:|---:|---:|
| Statistics | 0.372 | 0.296 | 0.407 | 0.349 |
| Signature | 0.371 | 0.281 | 0.422 | 0.343 |
| Combined | 0.372 | 0.296 | 0.417 | 0.352 |

### Causal detector on test paths

| Metric | Statistics | Signature | Combined | Combined - Statistics [95% CI] |
|---|---:|---:|---:|---:|
| Detector balanced accuracy | 0.3788 | 0.3791 | 0.3767 | -0.0021 [-0.0123, +0.0081] |
| Boundary F1 | 0.104 | 0.098 | 0.103 | -0.001 [-0.010, +0.008] |
| Detection probability within 100 | 0.252 | 0.190 | 0.264 | +0.012 [-0.021, +0.045] |
| Mean matched delay (obs.) | 47.2 | 51.2 | 49.3 | **+2.677 [+0.204, +5.151]** |
| False switches per 1,000 windows | 6.23 | 4.69 | 6.69 | +0.464 [-0.434, +1.361] |

### Balanced accuracy by observations since the last true regime change

| Bucket | Windows per replication | Statistics | Signature | Combined | Combined - Statistics [95% CI] |
|---|---:|---:|---:|---:|---:|
| 0-24 | 1112 | 0.3296 | 0.3299 | 0.3257 | -0.0039 [-0.0090, +0.0012] |
| 25-49 | 1105 | 0.3508 | 0.3543 | 0.3524 | +0.0016 [-0.0045, +0.0077] |
| 50-99 | 2088 | 0.3631 | 0.3624 | 0.3693 | **+0.0062 [+0.0015, +0.0109]** |
| 100+ | 20450 | 0.3600 | 0.3599 | 0.3630 | **+0.0030 [+0.0009, +0.0051]** |

## Logistic regression (second learner)

Selected regularization: statistics: C=0.01 x26, C=0.1 x6, C=1 x18; signature: C=0.01 x11, C=0.1 x16, C=1 x23; combined: C=0.01 x14, C=0.1 x10, C=1 x26.

### Pointwise classification

| Metric | Mean | 95% t CI | Paired Heston row |
|---|---:|---:|---:|
| Balanced accuracy, Statistics | 0.3518 | [0.3457, 0.3579] | 0.3978 |
| Balanced accuracy, Signature | 0.3667 | [0.3618, 0.3716] | 0.3953 |
| Balanced accuracy, Combined | 0.3687 | [0.3634, 0.3740] | 0.4166 |
| Combined minus Statistics (48 of 50 positive) | **+0.0169** | **[+0.0136, +0.0202]** | +0.0188 |
| Signature minus Statistics (37 of 50 positive) | +0.0149 | [+0.0088, +0.0211] | -0.0025 |

Per-regime test recall (mean over replications):

| Representation | Regime 0 | Regime 1 | Regime 2 | Macro F1 |
|---|---:|---:|---:|---:|
| Statistics | 0.368 | 0.291 | 0.397 | 0.344 |
| Signature | 0.451 | 0.248 | 0.401 | 0.354 |
| Combined | 0.404 | 0.292 | 0.410 | 0.361 |

### Causal detector on test paths

| Metric | Statistics | Signature | Combined | Combined - Statistics [95% CI] |
|---|---:|---:|---:|---:|
| Detector balanced accuracy | 0.3552 | 0.3843 | 0.3869 | **+0.0317 [+0.0216, +0.0418]** |
| Boundary F1 | 0.098 | 0.101 | 0.104 | +0.006 [-0.005, +0.017] |
| Detection probability within 100 | 0.279 | 0.256 | 0.312 | +0.034 [-0.018, +0.085] |
| Mean matched delay (obs.) | 49.4 | 51.7 | 52.5 | **+2.775 [+0.059, +5.490]** |
| False switches per 1,000 windows | 7.86 | 6.55 | 7.97 | +0.108 [-1.248, +1.464] |

### Balanced accuracy by observations since the last true regime change

| Bucket | Windows per replication | Statistics | Signature | Combined | Combined - Statistics [95% CI] |
|---|---:|---:|---:|---:|---:|
| 0-24 | 1112 | 0.3298 | 0.3128 | 0.3183 | **-0.0114 [-0.0224, -0.0004]** |
| 25-49 | 1105 | 0.3466 | 0.3602 | 0.3583 | **+0.0117 [+0.0006, +0.0229]** |
| 50-99 | 2088 | 0.3533 | 0.3748 | 0.3711 | **+0.0178 [+0.0100, +0.0257]** |
| 100+ | 20450 | 0.3532 | 0.3697 | 0.3722 | **+0.0189 [+0.0150, +0.0228]** |

## Learner contrast (paired, same windows)

| Representation | Logistic minus Random Forest balanced accuracy [95% CI] |
|---|---:|
| Statistics | **-0.0065 [-0.0111, -0.0019]** |
| Signature | **+0.0085 [+0.0050, +0.0119]** |
| Combined | **+0.0074 [+0.0028, +0.0120]** |

## Reading

- **All representations lose most of their discrimination**: Statistics 0.3583, Combined 0.3613, against
  0.3952 / 0.4027 in the Heston row (chance 0.333). With the rough driver the variance barely moves around
  theta (sd ratio 0.40 of Heston; return excess kurtosis 0.24 against 1.05), so a 50-return window
  carries little information about the mean-reversion speed. Balanced accuracy is flat across the
  time-since-change buckets (0.326 at 0-24 to 0.363 at 100+).
- **The design is no longer marginally matched.** The audit shows the three regimes share mean(V) but not
  sd(V) (0.0143 / 0.0189 / 0.0214); the fast regime, which has the largest sigma, keeps the most dispersion. Part of whatever
  separates the regimes here is therefore a level-dispersion difference, which the Heston row excludes by
  construction.
- **The forest's signature increment is small but excludes zero**: **+0.0030 [+0.0014, +0.0046]** (33 of 50; Heston row
  +0.0075 [+0.0053, +0.0097]); Signature alone -0.0000 [-0.0052, +0.0052]. Gains, where resolved, are in windows inside the new regime
  (**+0.0062 [+0.0015, +0.0109]** at 50-99, **+0.0030 [+0.0009, +0.0051]** at 100+).
- **The logistic learner**: **+0.0169 [+0.0136, +0.0202]** (48 of 50); logistic Combined versus forest Combined **+0.0074 [+0.0028, +0.0120]**.
  At H = 0.35 the Heston pattern reappears in weaker form: the logistic learner extracts the larger increment (48 of 50 positive) and its Combined model is the most accurate cell of the row (0.3687); as in the Heston row the gain is negative at 0-24 (**-0.0114 [-0.0224, -0.0004]**).
- **Event level is noisy at this accuracy level** (boundary F1 about 0.10, detection probability about 0.26):
  both learners' Combined models match changes about 2.7 observations *later* than their Statistics models (**+2.68 [+0.20, +5.15]** forest, **+2.77 [+0.06, +5.49]** logistic), with detector balanced accuracy up only for the logistic learner (**+0.0317 [+0.0216, +0.0418]**); nothing else excludes zero.
- **Comparability caveat.** Same parameter families as the Heston row under a different noise; the row does not
  isolate roughness from the marginal change documented in the audit.

## Output files

`config.json`, `variance_audit.csv`, `replications.csv`, `summary.csv`, `class_counts.csv`, `test_diagnostics.csv`
(per learner and representation: recalls, confusion matrices, detector metrics and operating points, stratified BA,
selected C).
