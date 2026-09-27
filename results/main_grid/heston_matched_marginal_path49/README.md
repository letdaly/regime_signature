# Main grid, row 2: standard Heston, matched-marginal regimes

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

Design as in `results/experiment_b_diagnostics`: within each path all regimes share theta, mu, rho, and sigma^2/kappa; only kappa changes, so the stationary CIR law of the variance is identical across regimes and only its time scale differs. Four Euler substeps per observation. Classwise probability diagnostics and calibration bins are also saved.

## Pointwise classification

| Metric | Mean | 95% t CI | frozen 10/3/5 raw |
|---|---:|---:|---:|
| Balanced accuracy, Statistics | 0.3952 | [0.3896, 0.4007] | 0.4007 |
| Balanced accuracy, Signature | 0.3990 | [0.3930, 0.4050] | 0.3970 |
| Balanced accuracy, Combined | 0.4018 | [0.3963, 0.4073] | 0.4067 |
| Combined minus Statistics (44 of 50 positive) | **+0.0066** | **[+0.0045, +0.0087]** | +0.0060 |
| Signature minus Statistics (28 of 50 positive) | +0.0038 | [-0.0014, +0.0091] | -0.0037 |

Per-regime test recall (mean over replications):

| Representation | Regime 0 | Regime 1 | Regime 2 | Macro F1 |
|---|---:|---:|---:|---:|
| Statistics | 0.386 | 0.281 | 0.519 | 0.380 |
| Signature | 0.451 | 0.245 | 0.501 | 0.379 |
| Combined | 0.395 | 0.270 | 0.541 | 0.384 |

## Causal detector on test paths

| Metric | Statistics | Signature | Combined | Combined - Statistics [95% CI] |
|---|---:|---:|---:|---:|
| Detector balanced accuracy | 0.4151 | 0.4251 | 0.4201 | +0.0050 [-0.0056, +0.0156] |
| Boundary F1 | 0.114 | 0.117 | 0.115 | +0.002 [-0.006, +0.009] |
| Detection probability within 100 | 0.227 | 0.196 | 0.232 | +0.005 [-0.029, +0.039] |
| Mean matched delay (obs.) | 49.1 | 51.8 | 51.1 | +1.162 [-1.583, +3.906] |
| False switches per 1,000 windows | 4.93 | 3.73 | 4.97 | +0.037 [-0.816, +0.891] |

## Balanced accuracy by observations since the last true regime change

| Bucket | Windows per replication | Statistics | Signature | Combined | Combined - Statistics [95% CI] |
|---|---:|---:|---:|---:|---:|
| 0-24 | 1150 | 0.3177 | 0.3141 | 0.3158 | -0.0020 [-0.0069, +0.0030] |
| 25-49 | 1142 | 0.3778 | 0.3851 | 0.3839 | +0.0061 [-0.0003, +0.0125] |
| 50-99 | 2150 | 0.4014 | 0.4102 | 0.4135 | **+0.0121** **[+0.0072, +0.0171]** |
| 100+ | 20313 | 0.4001 | 0.4034 | 0.4065 | **+0.0065** **[+0.0042, +0.0087]** |

Fast-regime (regime 2) prediction share versus its prevalence:

| Representation | Predicted share | Prevalence | Frozen raw-signature share |
|---|---:|---:|---:|
| Statistics | 0.417 | 0.301 | 0.426 |
| Signature | 0.400 | 0.301 | 0.567 |
| Combined | 0.431 | 0.301 | 0.490 |

## Reading

- The Combined gain over Statistics is **+0.0066** [+0.0045, +0.0087], positive in 44 of 50 replications, essentially the same as with raw signatures (+0.0060). Signature alone is now on par with Statistics (+0.0038, CI includes zero) instead of below it (-0.0037 with raw signatures).
- The raw signature's fast-regime bias is gone. With raw signatures the Signature model predicted the fast regime in 56.7% of windows against a 32.9% prevalence and bought its fast recall (0.661) with slow recall of 0.304. With log-signatures the predicted fast share is 0.400 (prevalence 0.301), fast recall 0.501, and slow recall 0.451 - higher than Statistics (0.386). The stable signature effect in the frozen study, a fast-time-scale preference, was therefore largely a property of the raw representation, whose higher-order coordinates scale with realized variance; the log-signature gain is spread across the slow and fast regimes rather than concentrated in one class.
- Event-level metrics remain a null result: boundary F1 about 0.11 for all representations, detection probability about 0.23, delay about 50 observations, and every paired detector interval includes zero.
- Stratified by time since the last change, balanced accuracy is flat at about 0.40 from 25 observations onward - in a matched-marginal contrast a mature window is not easier than a fresh one, because the variance level carries no regime information and a 50-observation window sees the same amount of dynamics either way. The Combined gain is concentrated in windows lying entirely inside the new regime: 50-99 (+0.0121, excludes zero) and 100+ (+0.0065, excludes zero); nothing in 0-24.

## Output files

`config.json`, `replications.csv`, `summary.csv`, `class_counts.csv`, `test_diagnostics.csv`, `classwise_metrics.csv`, `classwise_summary.csv`, `classwise_contrasts.csv`, `calibration_bins.csv`.
