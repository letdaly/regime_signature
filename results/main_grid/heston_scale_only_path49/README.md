# Main grid, row 1: standard Heston, scale-only regimes

> **Superseded on 2026-09-19.** This run used the earlier signature path
> (W points starting at (0, r_1, |r_1|), so the first return of every window
> never reached the signature; `signature_path` absent from `config.json`).
> The origin-anchored path with W increments and the added logistic learner
> are in the directory without the `_path49` suffix. Kept for the record; the
> Statistics columns are unaffected by the path convention.

## Configuration

Main-grid configuration (2026-09-18): 50 paired replications, 10 train / **10 validation** / 5 test paths,
5,000 observations per path, 50-step causal windows, Random Forest (200 trees, depth 6, balanced),
**level-3 log-signatures** (14 coordinates) on the three-channel path, Statistics 11 features, Combined 25.
Causal detector tuned on validation paths only (27-point grid), event tolerance 100.
Base seed unchanged from the frozen study, so replication k draws the same parameter families;
validation and test paths differ because ten validation paths are drawn before the test paths.

Design as in `results/experiment_a_scale_only`: within each path all regimes share mu, kappa, rho, and c; only theta_i changes, sigma_i = c sqrt(kappa theta_i).

## Pointwise classification

| Metric | Mean | 95% t CI | frozen 10/3/5 raw |
|---|---:|---:|---:|
| Balanced accuracy, Statistics | 0.7353 | [0.7275, 0.7431] | 0.7392 |
| Balanced accuracy, Signature | 0.7300 | [0.7221, 0.7378] | 0.7442 |
| Balanced accuracy, Combined | 0.7380 | [0.7303, 0.7458] | 0.7466 |
| Combined minus Statistics (44 of 50 positive) | **+0.0027** | **[+0.0021, +0.0033]** | +0.0074 |
| Signature minus Statistics (13 of 50 positive) | **-0.0054** | **[-0.0075, -0.0032]** | +0.0050 |

Per-regime test recall (mean over replications):

| Representation | Regime 0 | Regime 1 | Regime 2 | Macro F1 |
|---|---:|---:|---:|---:|
| Statistics | 0.781 | 0.702 | 0.723 | 0.734 |
| Signature | 0.772 | 0.701 | 0.717 | 0.729 |
| Combined | 0.782 | 0.708 | 0.725 | 0.737 |

## Causal detector on test paths

| Metric | Statistics | Signature | Combined | Combined - Statistics [95% CI] |
|---|---:|---:|---:|---:|
| Detector balanced accuracy | 0.7338 | 0.7014 | 0.7359 | **+0.0021** **[+0.0005, +0.0036]** |
| Boundary F1 | 0.270 | 0.262 | 0.271 | +0.001 [-0.004, +0.006] |
| Detection probability within 100 | 0.488 | 0.433 | 0.492 | +0.004 [-0.008, +0.016] |
| Mean matched delay (obs.) | 64.4 | 65.3 | 63.2 | **-1.150** **[-2.004, -0.296]** |
| False switches per 1,000 windows | 4.01 | 3.59 | 4.03 | +0.023 [-0.131, +0.178] |

## Balanced accuracy by observations since the last true regime change

| Bucket | Windows per replication | Statistics | Signature | Combined | Combined - Statistics [95% CI] |
|---|---:|---:|---:|---:|---:|
| 0-24 | 1155 | 0.1106 | 0.1169 | 0.1102 | -0.0004 [-0.0024, +0.0016] |
| 25-49 | 1150 | 0.1896 | 0.1964 | 0.1906 | +0.0011 [-0.0014, +0.0036] |
| 50-99 | 2143 | 0.4467 | 0.4502 | 0.4487 | +0.0020 [-0.0004, +0.0045] |
| 100+ | 20307 | 0.8346 | 0.8271 | 0.8378 | **+0.0031** **[+0.0024, +0.0038]** |

## Reading

- With log-signatures the Combined gain over Statistics shrinks from +0.0074 (raw, 39 coordinates) to **+0.0027**, still excluding zero and positive in 44 of 50 replications. Signature alone is now *below* Statistics (-0.0054): in a pure scale contrast the 14 log-signature coordinates carry less of the variance-level information than the 39 raw coordinates, whose higher-order terms scale with realized variance.
- The detector picture is the same as in the frozen study but cleaner: detector balanced accuracy +0.0021 (excludes zero), mean delay about one observation shorter (-1.15, excludes zero), boundary F1 and detection probability unchanged. With ten validation paths the false-switch penalty seen in the 3-path run (+0.26 per 1,000) disappears (+0.02, CI includes zero), consistent with better-selected operating points.
- Stratified by time since the last true change, the Combined gain lives entirely in mature windows (100+: +0.0031, excludes zero). In the three transition buckets the interval includes zero. Balanced accuracy climbs 0.11 -> 0.19 -> 0.45 -> 0.83: windows that lie entirely in the new regime but within 50-99 observations of the change are still far below mature windows, because the variance is continuous at a switch and needs about 1/kappa to reach the new theta. Mean delays of about 64 observations match this transient.

## Output files

`config.json`, `replications.csv`, `summary.csv`, `class_counts.csv`, `test_diagnostics.csv` (per-representation recalls, confusion matrices, detector metrics and operating points, stratified BA).
