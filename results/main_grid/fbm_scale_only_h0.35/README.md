# Main grid, row 3: fBM-driven stochastic volatility, scale-only regimes, H = 0.35

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
`results/main_grid/heston_scale_only` (`sigma_scale = 1`, 1 substep per observation), so
each replication here is the same family under rough variance noise. The Gamma stationary initialisation is exact
only at H = 0.5; the audit below quantifies the resulting marginal mismatch (identical families, mismatch stated as a limitation).

## Stationary variance audit (first replication's family, 10,000 no-switch observations)

| Setting | Regime | kappa | sigma | mean(V) | sd(V) | CIR sd | sd ratio to Heston | Return excess kurtosis | acf1(dV) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rough | 0 | 4.05 | 0.149 | 0.0188 | 0.0037 | 0.0073 | 0.56 | 0.09 | -0.20 |
| rough | 1 | 4.05 | 0.327 | 0.0901 | 0.0184 | 0.0352 | 0.52 | 0.11 | -0.20 |
| rough | 2 | 4.05 | 0.410 | 0.1394 | 0.0265 | 0.0555 | 0.48 | 0.12 | -0.19 |
| heston_reference | 0 | 4.05 | 0.149 | 0.0190 | 0.0066 | 0.0073 | - | 0.42 | +0.01 |
| heston_reference | 1 | 4.05 | 0.327 | 0.0935 | 0.0355 | 0.0352 | - | 0.35 | -0.00 |
| heston_reference | 2 | 4.05 | 0.410 | 0.1511 | 0.0555 | 0.0555 | - | 0.72 | +0.02 |

## Random Forest (main-table learner)

### Pointwise classification

| Metric | Mean | 95% t CI | Paired Heston row |
|---|---:|---:|---:|
| Balanced accuracy, Statistics | 0.7874 | [0.7802, 0.7945] | 0.7353 |
| Balanced accuracy, Signature | 0.7808 | [0.7734, 0.7883] | 0.7306 |
| Balanced accuracy, Combined | 0.7887 | [0.7815, 0.7958] | 0.7382 |
| Combined minus Statistics (39 of 50 positive) | **+0.0013** | **[+0.0008, +0.0019]** | +0.0029 |
| Signature minus Statistics (2 of 50 positive) | -0.0065 | [-0.0078, -0.0053] | -0.0047 |

Per-regime test recall (mean over replications):

| Representation | Regime 0 | Regime 1 | Regime 2 | Macro F1 |
|---|---:|---:|---:|---:|
| Statistics | 0.792 | 0.808 | 0.762 | 0.789 |
| Signature | 0.784 | 0.805 | 0.753 | 0.782 |
| Combined | 0.793 | 0.811 | 0.762 | 0.790 |

### Causal detector on test paths

| Metric | Statistics | Signature | Combined | Combined - Statistics [95% CI] |
|---|---:|---:|---:|---:|
| Detector balanced accuracy | 0.7807 | 0.7731 | 0.7817 | **+0.0010 [+0.0000, +0.0019]** |
| Boundary F1 | 0.315 | 0.312 | 0.318 | +0.003 [-0.003, +0.008] |
| Detection probability within 100 | 0.526 | 0.505 | 0.525 | -0.001 [-0.011, +0.009] |
| Mean matched delay (obs.) | 67.1 | 68.4 | 67.2 | +0.087 [-0.599, +0.773] |
| False switches per 1,000 windows | 3.40 | 3.26 | 3.31 | -0.085 [-0.203, +0.033] |

### Balanced accuracy by observations since the last true regime change

| Bucket | Windows per replication | Statistics | Signature | Combined | Combined - Statistics [95% CI] |
|---|---:|---:|---:|---:|---:|
| 0-24 | 1148 | 0.0703 | 0.0748 | 0.0720 | +0.0017 [-0.0000, +0.0034] |
| 25-49 | 1142 | 0.1374 | 0.1428 | 0.1435 | **+0.0062 [+0.0029, +0.0094]** |
| 50-99 | 2153 | 0.4308 | 0.4279 | 0.4335 | **+0.0027 [+0.0004, +0.0050]** |
| 100+ | 20312 | 0.9076 | 0.8994 | 0.9085 | **+0.0009 [+0.0004, +0.0015]** |

## Logistic regression (second learner)

Selected regularization: statistics: C=0.01 x16, C=0.1 x21, C=1 x13; signature: C=0.01 x25, C=0.1 x11, C=1 x14; combined: C=0.01 x17, C=0.1 x22, C=1 x11.

### Pointwise classification

| Metric | Mean | 95% t CI | Paired Heston row |
|---|---:|---:|---:|
| Balanced accuracy, Statistics | 0.7811 | [0.7740, 0.7883] | 0.7303 |
| Balanced accuracy, Signature | 0.7781 | [0.7715, 0.7846] | 0.7355 |
| Balanced accuracy, Combined | 0.7847 | [0.7777, 0.7916] | 0.7385 |
| Combined minus Statistics (43 of 50 positive) | **+0.0035** | **[+0.0025, +0.0046]** | +0.0083 |
| Signature minus Statistics (18 of 50 positive) | -0.0030 | [-0.0054, -0.0007] | +0.0052 |

Per-regime test recall (mean over replications):

| Representation | Regime 0 | Regime 1 | Regime 2 | Macro F1 |
|---|---:|---:|---:|---:|
| Statistics | 0.806 | 0.802 | 0.735 | 0.783 |
| Signature | 0.837 | 0.738 | 0.760 | 0.777 |
| Combined | 0.814 | 0.801 | 0.739 | 0.786 |

### Causal detector on test paths

| Metric | Statistics | Signature | Combined | Combined - Statistics [95% CI] |
|---|---:|---:|---:|---:|
| Detector balanced accuracy | 0.7770 | 0.7693 | 0.7828 | **+0.0057 [+0.0033, +0.0082]** |
| Boundary F1 | 0.313 | 0.331 | 0.335 | **+0.023 [+0.014, +0.032]** |
| Detection probability within 100 | 0.511 | 0.556 | 0.565 | **+0.055 [+0.037, +0.072]** |
| Mean matched delay (obs.) | 67.7 | 66.3 | 66.2 | **-1.497 [-2.762, -0.231]** |
| False switches per 1,000 windows | 3.28 | 3.37 | 3.40 | +0.117 [-0.071, +0.305] |

### Balanced accuracy by observations since the last true regime change

| Bucket | Windows per replication | Statistics | Signature | Combined | Combined - Statistics [95% CI] |
|---|---:|---:|---:|---:|---:|
| 0-24 | 1148 | 0.0777 | 0.0833 | 0.0780 | +0.0003 [-0.0023, +0.0029] |
| 25-49 | 1142 | 0.1549 | 0.2017 | 0.1841 | **+0.0292 [+0.0248, +0.0335]** |
| 50-99 | 2153 | 0.4464 | 0.5044 | 0.4791 | **+0.0327 [+0.0281, +0.0374]** |
| 100+ | 20312 | 0.8969 | 0.8836 | 0.8960 | -0.0009 [-0.0021, +0.0002] |

## Learner contrast (paired, same windows)

| Representation | Logistic minus Random Forest balanced accuracy [95% CI] |
|---|---:|
| Statistics | **-0.0062 [-0.0084, -0.0041]** |
| Signature | -0.0028 [-0.0056, +0.0001] |
| Combined | **-0.0040 [-0.0063, -0.0018]** |

## Reading

- **Rough noise makes the scale contrast easier, not harder, for every representation.** Statistics alone reaches
  0.7874 against 0.7353 in the Heston row on the same parameter families: antipersistent fGn shrinks the
  stationary dispersion of the variance (sd ratio 0.52 of Heston in the audit) so the variance *level* is a
  cleaner regime marker. Mature-window balanced accuracy is 0.909.
- **The forest's signature increment shrinks but stays positive**: **+0.0013 [+0.0008, +0.0019]** (39 of 50) against +0.0029 [+0.0022, +0.0036] in
  the Heston row; Signature alone is below Statistics (-0.0065 [-0.0078, -0.0053]). Unlike the Heston row, the gain now sits in
  the transition buckets (25-49: **+0.0062 [+0.0029, +0.0094]**; 50-99: **+0.0027 [+0.0004, +0.0050]**) rather than in mature windows (**+0.0009 [+0.0004, +0.0015]**): once the
  level has settled there is nothing left to add, and the signature helps while the window straddles the switch.
- **The logistic learner gains more within itself (+0.0035 [+0.0025, +0.0046], 43 of 50) but ends below the forest** (Combined
  -0.0040 [-0.0063, -0.0018]); its gain is again in the 25-99 buckets (**+0.0292 [+0.0248, +0.0335]**, **+0.0327 [+0.0281, +0.0374]**) and slightly negative in mature
  windows (-0.0009 [-0.0021, +0.0002]).
- **Event level.** No forest detector contrast excludes zero except, marginally, detector balanced accuracy at
  H = 0.35. For the logistic learner boundary F1 **+0.023 [+0.014, +0.032]**, detection probability **+0.055 [+0.037, +0.072]** and delay **-1.50 [-2.76, -0.23]**
  exclude zero with false switches unchanged (+0.12 [-0.07, +0.31]), reproducing the Heston scale-only pattern.
- **Comparability caveat.** This row is the same parameter family under a different noise, not a marginally
  matched rough model: variance dispersion and return kurtosis are far below the Heston values (audit table).
  The H effect on the table therefore mixes roughness with a marginal change, and the row answers "does the
  fBM-increment approximation change the answer" rather than "does roughness per se".

## Output files

`config.json`, `variance_audit.csv`, `replications.csv`, `summary.csv`, `class_counts.csv`, `test_diagnostics.csv`
(per learner and representation: recalls, confusion matrices, detector metrics and operating points, stratified BA,
selected C).
