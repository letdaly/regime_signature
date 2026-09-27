# Main grid, row 3: fBM-driven stochastic volatility, scale-only regimes, H = 0.2

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
`results/main_grid/heston_scale_only` (`sigma_scale = 1`, 1 substep per observation), so
each replication here is the same family under rough variance noise. The Gamma stationary initialisation is exact
only at H = 0.5; the audit below quantifies the resulting marginal mismatch (identical families, mismatch stated as a limitation).

## Stationary variance audit (first replication's family, 10,000 no-switch observations)

| Setting | Regime | kappa | sigma | mean(V) | sd(V) | CIR sd | sd ratio to Heston | Return excess kurtosis | acf1(dV) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rough | 0 | 4.05 | 0.149 | 0.0184 | 0.0019 | 0.0073 | 0.29 | 0.01 | -0.35 |
| rough | 1 | 4.05 | 0.327 | 0.0880 | 0.0095 | 0.0352 | 0.27 | 0.02 | -0.35 |
| rough | 2 | 4.05 | 0.410 | 0.1384 | 0.0146 | 0.0555 | 0.26 | 0.03 | -0.34 |
| heston_reference | 0 | 4.05 | 0.149 | 0.0190 | 0.0066 | 0.0073 | - | 0.42 | +0.01 |
| heston_reference | 1 | 4.05 | 0.327 | 0.0935 | 0.0355 | 0.0352 | - | 0.35 | -0.00 |
| heston_reference | 2 | 4.05 | 0.410 | 0.1511 | 0.0555 | 0.0555 | - | 0.72 | +0.02 |

## Random Forest (main-table learner)

### Pointwise classification

| Metric | Mean | 95% t CI | Paired Heston row |
|---|---:|---:|---:|
| Balanced accuracy, Statistics | 0.8024 | [0.7956, 0.8091] | 0.7353 |
| Balanced accuracy, Signature | 0.7960 | [0.7889, 0.8032] | 0.7306 |
| Balanced accuracy, Combined | 0.8035 | [0.7967, 0.8102] | 0.7382 |
| Combined minus Statistics (39 of 50 positive) | **+0.0011** | **[+0.0008, +0.0014]** | +0.0029 |
| Signature minus Statistics (2 of 50 positive) | -0.0063 | [-0.0075, -0.0051] | -0.0047 |

Per-regime test recall (mean over replications):

| Representation | Regime 0 | Regime 1 | Regime 2 | Macro F1 |
|---|---:|---:|---:|---:|
| Statistics | 0.798 | 0.828 | 0.781 | 0.804 |
| Signature | 0.790 | 0.828 | 0.770 | 0.798 |
| Combined | 0.799 | 0.831 | 0.781 | 0.805 |

### Causal detector on test paths

| Metric | Statistics | Signature | Combined | Combined - Statistics [95% CI] |
|---|---:|---:|---:|---:|
| Detector balanced accuracy | 0.7931 | 0.7858 | 0.7935 | +0.0005 [-0.0003, +0.0012] |
| Boundary F1 | 0.353 | 0.335 | 0.353 | +0.000 [-0.005, +0.005] |
| Detection probability within 100 | 0.556 | 0.518 | 0.557 | +0.001 [-0.009, +0.011] |
| Mean matched delay (obs.) | 69.6 | 70.0 | 69.6 | -0.029 [-0.595, +0.538] |
| False switches per 1,000 windows | 2.98 | 2.94 | 2.98 | +0.000 [-0.078, +0.078] |

### Balanced accuracy by observations since the last true regime change

| Bucket | Windows per replication | Statistics | Signature | Combined | Combined - Statistics [95% CI] |
|---|---:|---:|---:|---:|---:|
| 0-24 | 1148 | 0.0574 | 0.0608 | 0.0584 | +0.0010 [-0.0003, +0.0023] |
| 25-49 | 1142 | 0.1206 | 0.1254 | 0.1302 | **+0.0095 [+0.0072, +0.0119]** |
| 50-99 | 2153 | 0.4273 | 0.4260 | 0.4311 | **+0.0038 [+0.0017, +0.0060]** |
| 100+ | 20312 | 0.9282 | 0.9201 | 0.9286 | +0.0004 [-0.0000, +0.0007] |

## Logistic regression (second learner)

Selected regularization: statistics: C=0.01 x28, C=0.1 x15, C=1 x7; signature: C=0.01 x29, C=0.1 x9, C=1 x12; combined: C=0.01 x21, C=0.1 x19, C=1 x10.

### Pointwise classification

| Metric | Mean | 95% t CI | Paired Heston row |
|---|---:|---:|---:|
| Balanced accuracy, Statistics | 0.7955 | [0.7886, 0.8025] | 0.7303 |
| Balanced accuracy, Signature | 0.7904 | [0.7841, 0.7968] | 0.7355 |
| Balanced accuracy, Combined | 0.7973 | [0.7905, 0.8041] | 0.7385 |
| Combined minus Statistics (34 of 50 positive) | **+0.0018** | **[+0.0007, +0.0028]** | +0.0083 |
| Signature minus Statistics (11 of 50 positive) | -0.0051 | [-0.0071, -0.0031] | +0.0052 |

Per-regime test recall (mean over replications):

| Representation | Regime 0 | Regime 1 | Regime 2 | Macro F1 |
|---|---:|---:|---:|---:|
| Statistics | 0.811 | 0.820 | 0.755 | 0.797 |
| Signature | 0.838 | 0.759 | 0.774 | 0.790 |
| Combined | 0.816 | 0.820 | 0.756 | 0.799 |

### Causal detector on test paths

| Metric | Statistics | Signature | Combined | Combined - Statistics [95% CI] |
|---|---:|---:|---:|---:|
| Detector balanced accuracy | 0.7914 | 0.7841 | 0.7937 | **+0.0023 [+0.0005, +0.0042]** |
| Boundary F1 | 0.350 | 0.349 | 0.360 | **+0.010 [+0.000, +0.020]** |
| Detection probability within 100 | 0.538 | 0.568 | 0.577 | **+0.038 [+0.019, +0.058]** |
| Mean matched delay (obs.) | 69.3 | 67.0 | 66.9 | **-2.348 [-3.409, -1.287]** |
| False switches per 1,000 windows | 2.88 | 3.17 | 3.04 | +0.162 [-0.035, +0.360] |

### Balanced accuracy by observations since the last true regime change

| Bucket | Windows per replication | Statistics | Signature | Combined | Combined - Statistics [95% CI] |
|---|---:|---:|---:|---:|---:|
| 0-24 | 1148 | 0.0637 | 0.0729 | 0.0662 | **+0.0026 [+0.0002, +0.0049]** |
| 25-49 | 1142 | 0.1408 | 0.1886 | 0.1725 | **+0.0317 [+0.0264, +0.0371]** |
| 50-99 | 2153 | 0.4461 | 0.5024 | 0.4778 | **+0.0318 [+0.0271, +0.0365]** |
| 100+ | 20312 | 0.9164 | 0.9005 | 0.9131 | **-0.0033 [-0.0045, -0.0022]** |

## Learner contrast (paired, same windows)

| Representation | Logistic minus Random Forest balanced accuracy [95% CI] |
|---|---:|
| Statistics | **-0.0068 [-0.0088, -0.0049]** |
| Signature | **-0.0056 [-0.0080, -0.0032]** |
| Combined | **-0.0061 [-0.0082, -0.0040]** |

## Reading

- **Rough noise makes the scale contrast easier, not harder, for every representation.** Statistics alone reaches
  0.8024 against 0.7353 in the Heston row on the same parameter families: antipersistent fGn shrinks the
  stationary dispersion of the variance (sd ratio 0.27 of Heston in the audit) so the variance *level* is a
  cleaner regime marker. Mature-window balanced accuracy is 0.929.
- **The forest's signature increment shrinks but stays positive**: **+0.0011 [+0.0008, +0.0014]** (39 of 50) against +0.0029 [+0.0022, +0.0036] in
  the Heston row; Signature alone is below Statistics (-0.0063 [-0.0075, -0.0051]). Unlike the Heston row, the gain now sits in
  the transition buckets (25-49: **+0.0095 [+0.0072, +0.0119]**; 50-99: **+0.0038 [+0.0017, +0.0060]**) rather than in mature windows (+0.0004 [-0.0000, +0.0007]): once the
  level has settled there is nothing left to add, and the signature helps while the window straddles the switch.
- **The logistic learner gains more within itself (+0.0018 [+0.0007, +0.0028], 34 of 50) but ends below the forest** (Combined
  -0.0061 [-0.0082, -0.0040]); its gain is again in the 25-99 buckets (**+0.0317 [+0.0264, +0.0371]**, **+0.0318 [+0.0271, +0.0365]**) and slightly negative in mature
  windows (**-0.0033 [-0.0045, -0.0022]**).
- **Event level.** No forest detector contrast excludes zero except, marginally, detector balanced accuracy at
  H = 0.35. For the logistic learner boundary F1 **+0.010 [+0.000, +0.020]**, detection probability **+0.038 [+0.019, +0.058]** and delay **-2.35 [-3.41, -1.29]**
  exclude zero with false switches unchanged (+0.16 [-0.04, +0.36]), reproducing the Heston scale-only pattern.
- **Comparability caveat.** This row is the same parameter family under a different noise, not a marginally
  matched rough model: variance dispersion and return kurtosis are far below the Heston values (audit table).
  The H effect on the table therefore mixes roughness with a marginal change, and the row answers "does the
  fBM-increment approximation change the answer" rather than "does roughness per se".

## Output files

`config.json`, `variance_audit.csv`, `replications.csv`, `summary.csv`, `class_counts.csv`, `test_diagnostics.csv`
(per learner and representation: recalls, confusion matrices, detector metrics and operating points, stratified BA,
selected C).
