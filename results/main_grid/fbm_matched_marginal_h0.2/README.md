# Main grid, row 3: fBM-driven stochastic volatility, matched-marginal regimes, H = 0.2

## Configuration

Run 2026-09-20: 50 paired replications, 10 train / 10 validation / 5 test paths, 5,000 observations per path,
50-step causal windows, level-3 log-signatures (14 coordinates) on the origin-anchored three-channel path
(`signature_path = origin_anchored_w_increments`), Statistics 11 features, Combined 25. Two learners on the same
column subsets: the fixed Random Forest (200 trees, depth 6, balanced; bare column names) and a multinomial
logistic regression (robust scaling and tail clipping fitted on training windows, C in {0.01, 0.1, 1} selected
by validation balanced accuracy; `logistic_` columns). Causal detector tuned on validation paths only (27-point
grid) for every learner-representation cell, event tolerance 100.

Simulator: `FBMStochasticVolatilitySimulator` (exact Hosking fGn with Hurst index 0.2 replacing the Brownian
increments of the CIR variance equation; the price shock uses the fGn *innovation*, so log returns stay serially
uncorrelated). Parameter families, base seed and replication seeds are identical to the paired Heston row
`results/main_grid/heston_matched_marginal` (`sigma_scale = 1`, 4 substeps per observation), so
each replication here is the same family under rough variance noise. The Gamma stationary initialisation is exact
only at H = 0.5; the audit below quantifies the resulting marginal mismatch (identical families, mismatch stated as a limitation).

## Stationary variance audit (first replication's family, 10,000 no-switch observations)

| Setting | Regime | kappa | sigma | mean(V) | sd(V) | CIR sd | sd ratio to Heston | Return excess kurtosis | acf1(dV) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rough | 0 | 0.83 | 0.210 | 0.0674 | 0.0047 | 0.0460 | 0.09 | 0.09 | -0.35 |
| rough | 1 | 4.61 | 0.495 | 0.0666 | 0.0082 | 0.0460 | 0.18 | 0.16 | -0.34 |
| rough | 2 | 14.88 | 0.890 | 0.0670 | 0.0113 | 0.0460 | 0.27 | 0.04 | -0.34 |
| heston_reference | 0 | 0.83 | 0.210 | 0.0785 | 0.0526 | 0.0460 | - | 1.54 | +0.01 |
| heston_reference | 1 | 4.61 | 0.495 | 0.0784 | 0.0450 | 0.0460 | - | 0.86 | -0.00 |
| heston_reference | 2 | 14.88 | 0.890 | 0.0773 | 0.0419 | 0.0460 | - | 0.76 | -0.03 |

## Random Forest (main-table learner)

### Pointwise classification

| Metric | Mean | 95% t CI | Paired Heston row |
|---|---:|---:|---:|
| Balanced accuracy, Statistics | 0.3430 | [0.3382, 0.3479] | 0.3952 |
| Balanced accuracy, Signature | 0.3376 | [0.3333, 0.3419] | 0.3998 |
| Balanced accuracy, Combined | 0.3456 | [0.3407, 0.3505] | 0.4027 |
| Combined minus Statistics (39 of 50 positive) | **+0.0025** | **[+0.0012, +0.0038]** | +0.0075 |
| Signature minus Statistics (17 of 50 positive) | -0.0055 | [-0.0099, -0.0010] | +0.0047 |

Per-regime test recall (mean over replications):

| Representation | Regime 0 | Regime 1 | Regime 2 | Macro F1 |
|---|---:|---:|---:|---:|
| Statistics | 0.336 | 0.307 | 0.386 | 0.333 |
| Signature | 0.331 | 0.306 | 0.376 | 0.321 |
| Combined | 0.341 | 0.313 | 0.383 | 0.335 |

### Causal detector on test paths

| Metric | Statistics | Signature | Combined | Combined - Statistics [95% CI] |
|---|---:|---:|---:|---:|
| Detector balanced accuracy | 0.3462 | 0.3472 | 0.3479 | +0.0018 [-0.0063, +0.0099] |
| Boundary F1 | 0.085 | 0.071 | 0.091 | +0.006 [-0.003, +0.014] |
| Detection probability within 100 | 0.240 | 0.124 | 0.217 | -0.024 [-0.055, +0.007] |
| Mean matched delay (obs.) | 50.0 | 47.4 | 49.1 | +0.224 [-2.808, +3.256] |
| False switches per 1,000 windows | 7.29 | 3.76 | 6.18 | **-1.102 [-2.072, -0.132]** |

### Balanced accuracy by observations since the last true regime change

| Bucket | Windows per replication | Statistics | Signature | Combined | Combined - Statistics [95% CI] |
|---|---:|---:|---:|---:|---:|
| 0-24 | 1112 | 0.3231 | 0.3227 | 0.3254 | +0.0024 [-0.0036, +0.0083] |
| 25-49 | 1105 | 0.3352 | 0.3384 | 0.3419 | **+0.0068 [+0.0010, +0.0126]** |
| 50-99 | 2088 | 0.3451 | 0.3412 | 0.3504 | +0.0053 [-0.0003, +0.0108] |
| 100+ | 20450 | 0.3444 | 0.3380 | 0.3464 | **+0.0021 [+0.0006, +0.0035]** |

## Logistic regression (second learner)

Selected regularization: statistics: C=0.01 x28, C=0.1 x7, C=1 x15; signature: C=0.01 x17, C=0.1 x10, C=1 x23; combined: C=0.01 x23, C=0.1 x8, C=1 x19.

### Pointwise classification

| Metric | Mean | 95% t CI | Paired Heston row |
|---|---:|---:|---:|
| Balanced accuracy, Statistics | 0.3421 | [0.3364, 0.3479] | 0.3978 |
| Balanced accuracy, Signature | 0.3437 | [0.3391, 0.3484] | 0.3953 |
| Balanced accuracy, Combined | 0.3475 | [0.3423, 0.3527] | 0.4166 |
| Combined minus Statistics (33 of 50 positive) | **+0.0054** | **[+0.0026, +0.0082]** | +0.0188 |
| Signature minus Statistics (28 of 50 positive) | +0.0016 | [-0.0045, +0.0076] | -0.0025 |

Per-regime test recall (mean over replications):

| Representation | Regime 0 | Regime 1 | Regime 2 | Macro F1 |
|---|---:|---:|---:|---:|
| Statistics | 0.343 | 0.304 | 0.379 | 0.334 |
| Signature | 0.369 | 0.282 | 0.381 | 0.333 |
| Combined | 0.352 | 0.310 | 0.381 | 0.341 |

### Causal detector on test paths

| Metric | Statistics | Signature | Combined | Combined - Statistics [95% CI] |
|---|---:|---:|---:|---:|
| Detector balanced accuracy | 0.3408 | 0.3487 | 0.3522 | **+0.0114 [+0.0027, +0.0201]** |
| Boundary F1 | 0.095 | 0.088 | 0.097 | +0.002 [-0.006, +0.010] |
| Detection probability within 100 | 0.254 | 0.219 | 0.327 | **+0.073 [+0.032, +0.115]** |
| Mean matched delay (obs.) | 50.2 | 49.1 | 48.4 | -1.774 [-4.760, +1.211] |
| False switches per 1,000 windows | 7.22 | 6.48 | 9.24 | **+2.019 [+0.841, +3.197]** |

### Balanced accuracy by observations since the last true regime change

| Bucket | Windows per replication | Statistics | Signature | Combined | Combined - Statistics [95% CI] |
|---|---:|---:|---:|---:|---:|
| 0-24 | 1112 | 0.3259 | 0.3151 | 0.3220 | -0.0039 [-0.0131, +0.0052] |
| 25-49 | 1105 | 0.3367 | 0.3460 | 0.3414 | +0.0047 [-0.0064, +0.0159] |
| 50-99 | 2088 | 0.3423 | 0.3472 | 0.3519 | **+0.0096 [+0.0028, +0.0164]** |
| 100+ | 20450 | 0.3435 | 0.3454 | 0.3491 | **+0.0056 [+0.0025, +0.0087]** |

## Learner contrast (paired, same windows)

| Representation | Logistic minus Random Forest balanced accuracy [95% CI] |
|---|---:|
| Statistics | -0.0009 [-0.0053, +0.0035] |
| Signature | **+0.0061 [+0.0025, +0.0098]** |
| Combined | +0.0019 [-0.0022, +0.0061] |

## Reading

- **All representations lose most of their discrimination**: Statistics 0.3430, Combined 0.3456, against
  0.3952 / 0.4027 in the Heston row (chance 0.333). With the rough driver the variance barely moves around
  theta (sd ratio 0.18 of Heston; return excess kurtosis 0.09 against 1.05), so a 50-return window
  carries little information about the mean-reversion speed. Balanced accuracy is flat across the
  time-since-change buckets (0.325 at 0-24 to 0.346 at 100+).
- **The design is no longer marginally matched.** The audit shows the three regimes share mean(V) but not
  sd(V) (0.0047 / 0.0082 / 0.0113); the fast regime, which has the largest sigma, keeps the most dispersion. Part of whatever
  separates the regimes here is therefore a level-dispersion difference, which the Heston row excludes by
  construction.
- **The forest's signature increment is small but excludes zero**: **+0.0025 [+0.0012, +0.0038]** (39 of 50; Heston row
  +0.0075 [+0.0053, +0.0097]); Signature alone -0.0055 [-0.0099, -0.0010]. Gains, where resolved, are in windows inside the new regime
  (+0.0053 [-0.0003, +0.0108] at 50-99, **+0.0021 [+0.0006, +0.0035]** at 100+).
- **The logistic learner**: **+0.0054 [+0.0026, +0.0082]** (33 of 50); logistic Combined versus forest Combined +0.0019 [-0.0022, +0.0061].
  At H = 0.2 the two learners are indistinguishable; the logistic gain is positive in only 33 of 50 replications.
- **Event level is noisy at this accuracy level** (boundary F1 about 0.09, detection probability about 0.22):
  the forest's Combined model makes fewer false switches than its Statistics model (**-1.10 [-2.07, -0.13]**) while the logistic learner's makes more (**+2.02 [+0.84, +3.20]**) and detects more changes (**+0.073 [+0.032, +0.115]**); neither pattern holds at H = 0.35.
- **Comparability caveat.** Same parameter families as the Heston row under a different noise; the row does not
  isolate roughness from the marginal change documented in the audit.

## Output files

`config.json`, `variance_audit.csv`, `replications.csv`, `summary.csv`, `class_counts.csv`, `test_diagnostics.csv`
(per learner and representation: recalls, confusion matrices, detector metrics and operating points, stratified BA,
selected C).
