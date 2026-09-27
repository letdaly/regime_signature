# Unified comparison grid

Both designs run 2026-09-21 with `experiment_unified_grid.py --jobs 12`: 50 paired replications, 10 / 10 / 5 training / validation / test paths of 5,000 observations, 50-return causal windows labelled by their endpoint state, origin-anchored three-channel path, level-3 signatures (raw 39 / log 14). Base seeds equal the main-grid rows' (`results/main_grid/heston_*`), so replication k simulates the same paths; the logistic Statistics and Statistics + log-sig cells reproduce the main grid to every digit. Every cell is a column subset of one feature matrix per replication; the three learners each select one setting on validation balanced accuracy from an equal three-point grid (forest depth 4 / 6 / 10 with 200 trees; logistic C 0.01 / 0.1 / 1 after robust scaling and clipping; histogram gradient boosting with 7 / 15 / 31 leaves, learning rate 0.05, 300 iterations; all class-balanced). The HMM block holds the forward-filtered state probabilities of a three-state Gaussian HMM fitted by EM on the pooled training paths (percent returns) and state-matched to labels on training windows; `HMM direct` is the balanced accuracy of its own argmax without a learner. Shuffled blocks recompute the signature after permuting the increments inside each window (raw and log share the permutation). Gains are in percentage points with 95% Student-t intervals over replications and the number of replications with a positive paired difference; bold marks intervals excluding zero. Every number is read from `<design>/summary.csv`.

## Design A (scale-only)

Test balanced accuracy (mean over 50 replications; chance 0.3333). HMM direct argmax without a learner: 0.7233 [0.7149, 0.7316].

| Cell (columns) | RF | Logit | GBM |
|---|---:|---:|---:|
| HMM filter block | 0.7264 | 0.7256 | 0.7263 |
| Statistics (11) | 0.7349 | 0.7303 | 0.7282 |
| Statistics + log-sig (25) | 0.7396 | 0.7385 | 0.7356 |
| Statistics + raw-sig (50) | 0.7435 | 0.7433 | 0.7366 |
| B3 (40) | 0.7378 | 0.7261 | 0.7284 |
| B3 + log-sig (54) | 0.7397 | 0.7310 | 0.7326 |
| B3 + raw-sig (79) | 0.7434 | 0.7365 | 0.7333 |
| log-sig alone (14) | 0.7360 | 0.7355 | 0.7409 |
| raw-sig alone (39) | 0.7420 | 0.7431 | 0.7403 |

Paired gains (pp):

| Contrast | RF | Logit | GBM |
|---|---:|---:|---:|
| Statistics -> + log-sig | **+0.47 [+0.39, +0.56] 49/50** | **+0.83 [+0.68, +0.97] 46/50** | **+0.74 [+0.63, +0.86] 49/50** |
| Statistics -> + raw-sig | **+0.86 [+0.73, +0.99] 50/50** | **+1.30 [+1.09, +1.52] 49/50** | **+0.85 [+0.71, +0.99] 49/50** |
| B3 -> + log-sig | **+0.19 [+0.11, +0.27] 37/50** | **+0.49 [+0.37, +0.61] 43/50** | **+0.42 [+0.34, +0.49] 48/50** |
| B3 -> + raw-sig | **+0.56 [+0.46, +0.67] 48/50** | **+1.04 [+0.87, +1.20] 48/50** | **+0.49 [+0.40, +0.57] 47/50** |
| raw-sig - log-sig (on Statistics) | **+0.39 [+0.28, +0.49] 43/50** | **+0.48 [+0.31, +0.64] 40/50** | **+0.10 [+0.02, +0.19] 30/50** |
| raw-sig - log-sig (on B3) | **+0.37 [+0.30, +0.45] 45/50** | **+0.54 [+0.40, +0.69] 44/50** | +0.07 [-0.01, +0.16] 29/50 |
| B3 - Statistics | **+0.28 [+0.18, +0.39] 40/50** | **-0.42 [-0.60, -0.24] 15/50** | +0.03 [-0.12, +0.17] 25/50 |
| HMM block - Statistics | **-0.85 [-1.06, -0.64] 4/50** | **-0.47 [-0.82, -0.13] 22/50** | -0.19 [-0.46, +0.08] 22/50 |

Order diagnostic (pp):

| Contrast | RF | Logit | GBM |
|---|---:|---:|---:|
| raw-sig alone: ordered - shuffled | **+0.69 [+0.60, +0.79] 50/50** | **+0.84 [+0.69, +1.00] 44/50** | **+0.61 [+0.46, +0.75] 45/50** |
| log-sig alone: ordered - shuffled | **+0.31 [+0.22, +0.41] 41/50** | **+0.75 [+0.55, +0.96] 43/50** | **+0.64 [+0.51, +0.77] 47/50** |
| Statistics + raw-sig: ordered - shuffled | **+0.52 [+0.43, +0.60] 49/50** | **+0.41 [+0.29, +0.54] 41/50** | **+0.52 [+0.42, +0.62] 47/50** |
| Statistics + log-sig: ordered - shuffled | **+0.24 [+0.16, +0.32] 39/50** | **+0.46 [+0.35, +0.57] 44/50** | **+0.45 [+0.36, +0.53] 46/50** |
| B3 + raw-sig: ordered - shuffled | **+0.50 [+0.42, +0.57] 50/50** | **+0.45 [+0.32, +0.59] 44/50** | **+0.51 [+0.42, +0.60] 48/50** |
| B3 + log-sig: ordered - shuffled | **+0.13 [+0.07, +0.20] 38/50** | **+0.51 [+0.39, +0.63] 42/50** | **+0.41 [+0.34, +0.49] 48/50** |
| Statistics -> + shuffled raw-sig | **+0.34 [+0.22, +0.46] 40/50** | **+0.89 [+0.71, +1.07] 45/50** | **+0.32 [+0.25, +0.40] 46/50** |
| Statistics -> + shuffled log-sig | **+0.23 [+0.16, +0.30] 41/50** | **+0.37 [+0.29, +0.45] 45/50** | **+0.30 [+0.23, +0.37] 44/50** |
| B3 -> + shuffled raw-sig | +0.07 [-0.01, +0.14] 33/50 | **+0.58 [+0.45, +0.72] 45/50** | -0.02 [-0.08, +0.04] 27/50 |
| B3 -> + shuffled log-sig | **+0.06 [+0.01, +0.10] 32/50** | -0.02 [-0.06, +0.02] 22/50 | +0.00 [-0.01, +0.01] 9/50 |

Learner contrasts on the same cell (pp):

| Cell | Logit - RF | GBM - RF |
|---|---:|---:|
| Statistics | **-0.46 [-0.71, -0.22] 12/50** | **-0.68 [-0.86, -0.50] 6/50** |
| Statistics + log-sig | -0.11 [-0.36, +0.15] 21/50 | **-0.40 [-0.56, -0.25] 10/50** |
| Statistics + raw-sig | -0.02 [-0.24, +0.20] 19/50 | **-0.69 [-0.89, -0.49] 6/50** |
| B3 | **-1.17 [-1.45, -0.89] 5/50** | **-0.94 [-1.16, -0.72] 6/50** |
| B3 + log-sig | **-0.87 [-1.17, -0.57] 8/50** | **-0.71 [-0.91, -0.51] 6/50** |
| B3 + raw-sig | **-0.69 [-0.95, -0.43] 9/50** | **-1.01 [-1.21, -0.81] 3/50** |

HMM filter fitted on training paths: state standard deviations (percent daily) 0.90 / 1.67 / 2.66 and persistence 0.996 / 0.991 / 0.993; the persistent initialization won in 44% of replications.

## Design B (matched-marginal)

Test balanced accuracy (mean over 50 replications; chance 0.3333). HMM direct argmax without a learner: 0.3443 [0.3367, 0.3518].

| Cell (columns) | RF | Logit | GBM |
|---|---:|---:|---:|
| HMM filter block | 0.3549 | 0.3480 | 0.3542 |
| Statistics (11) | 0.3948 | 0.3978 | 0.3928 |
| Statistics + log-sig (25) | 0.4063 | 0.4166 | 0.4072 |
| Statistics + raw-sig (50) | 0.4076 | 0.4431 | 0.4074 |
| B3 (40) | 0.4041 | 0.3929 | 0.3987 |
| B3 + log-sig (54) | 0.4076 | 0.4074 | 0.4034 |
| B3 + raw-sig (79) | 0.4082 | 0.4335 | 0.4040 |
| log-sig alone (14) | 0.4077 | 0.3953 | 0.4156 |
| raw-sig alone (39) | 0.4010 | 0.4394 | 0.4116 |

Paired gains (pp):

| Contrast | RF | Logit | GBM |
|---|---:|---:|---:|
| Statistics -> + log-sig | **+1.16 [+0.88, +1.44] 46/50** | **+1.88 [+1.59, +2.17] 49/50** | **+1.44 [+1.20, +1.68] 49/50** |
| Statistics -> + raw-sig | **+1.28 [+0.94, +1.62] 44/50** | **+4.53 [+4.07, +4.99] 50/50** | **+1.46 [+1.20, +1.71] 47/50** |
| B3 -> + log-sig | **+0.35 [+0.18, +0.53] 34/50** | **+1.45 [+1.23, +1.67] 48/50** | **+0.48 [+0.33, +0.63] 40/50** |
| B3 -> + raw-sig | **+0.41 [+0.22, +0.60] 35/50** | **+4.06 [+3.70, +4.42] 49/50** | **+0.53 [+0.37, +0.69] 39/50** |
| raw-sig - log-sig (on Statistics) | +0.12 [-0.12, +0.37] 26/50 | **+2.65 [+2.31, +2.99] 50/50** | +0.02 [-0.17, +0.20] 24/50 |
| raw-sig - log-sig (on B3) | +0.05 [-0.15, +0.25] 25/50 | **+2.60 [+2.34, +2.87] 50/50** | +0.05 [-0.09, +0.19] 24/50 |
| B3 - Statistics | **+0.93 [+0.63, +1.23] 41/50** | **-0.49 [-0.91, -0.06] 17/50** | **+0.59 [+0.32, +0.86] 38/50** |
| HMM block - Statistics | **-3.99 [-4.44, -3.54] 0/50** | **-4.98 [-5.59, -4.38] 0/50** | **-3.86 [-4.36, -3.37] 1/50** |

Order diagnostic (pp):

| Contrast | RF | Logit | GBM |
|---|---:|---:|---:|
| raw-sig alone: ordered - shuffled | **+1.54 [+1.25, +1.83] 46/50** | **+4.61 [+4.19, +5.04] 50/50** | **+2.83 [+2.42, +3.23] 48/50** |
| log-sig alone: ordered - shuffled | **+2.42 [+2.07, +2.76] 48/50** | **+4.66 [+4.17, +5.16] 50/50** | **+3.19 [+2.84, +3.54] 50/50** |
| Statistics + raw-sig: ordered - shuffled | **+0.70 [+0.52, +0.88] 44/50** | **+2.95 [+2.59, +3.32] 49/50** | **+1.06 [+0.86, +1.27] 45/50** |
| Statistics + log-sig: ordered - shuffled | **+0.70 [+0.48, +0.93] 43/50** | **+1.77 [+1.50, +2.03] 48/50** | **+1.16 [+0.96, +1.36] 47/50** |
| B3 + raw-sig: ordered - shuffled | **+0.32 [+0.14, +0.51] 39/50** | **+2.50 [+2.19, +2.81] 49/50** | **+0.55 [+0.40, +0.70] 42/50** |
| B3 + log-sig: ordered - shuffled | **+0.32 [+0.18, +0.45] 41/50** | **+1.45 [+1.23, +1.67] 48/50** | **+0.45 [+0.30, +0.60] 39/50** |
| Statistics -> + shuffled raw-sig | **+0.58 [+0.31, +0.86] 37/50** | **+1.58 [+1.25, +1.90] 47/50** | **+0.39 [+0.24, +0.55] 40/50** |
| Statistics -> + shuffled log-sig | **+0.45 [+0.24, +0.67] 37/50** | +0.11 [-0.02, +0.25] 29/50 | **+0.28 [+0.13, +0.42] 38/50** |
| B3 -> + shuffled raw-sig | +0.08 [-0.12, +0.28] 30/50 | **+1.56 [+1.32, +1.80] 47/50** | -0.02 [-0.17, +0.13] 24/50 |
| B3 -> + shuffled log-sig | +0.04 [-0.11, +0.18] 26/50 | +0.01 [-0.03, +0.04] 24/50 | +0.02 [-0.06, +0.10] 10/50 |

Learner contrasts on the same cell (pp):

| Cell | Logit - RF | GBM - RF |
|---|---:|---:|
| Statistics | +0.30 [-0.18, +0.79] 31/50 | -0.20 [-0.58, +0.19] 23/50 |
| Statistics + log-sig | **+1.03 [+0.54, +1.51] 38/50** | +0.09 [-0.23, +0.41] 28/50 |
| Statistics + raw-sig | **+3.55 [+3.15, +3.95] 50/50** | -0.02 [-0.35, +0.31] 25/50 |
| B3 | **-1.12 [-1.55, -0.68] 7/50** | **-0.54 [-0.90, -0.18] 17/50** |
| B3 + log-sig | -0.02 [-0.48, +0.44] 22/50 | **-0.42 [-0.75, -0.08] 21/50** |
| B3 + raw-sig | **+2.53 [+2.12, +2.95] 49/50** | **-0.42 [-0.75, -0.09] 19/50** |

HMM filter fitted on training paths: state standard deviations (percent daily) 1.66 / 1.74 / 1.65 and persistence 0.975 / 0.975 / 0.977; the persistent initialization won in 100% of replications.

## Output files

`<design>/config.json`, `replications.csv` (one wide row per replication: every cell's validation and test metrics, selected setting, contrasts, HMM diagnostics), `cells.csv` (tidy replication x learner x cell), `summary.csv` (levels, contrasts, positive counts). `probabilities/replication_NN.npz` holds the uint16-quantized validation and test probabilities of the seven Table-1 cells for every learner, with labels, path ids and window end times; `paths/replication_NN.npz` holds the log returns and regime labels of every simulated path. The two archives (about 470 MB per design) are git-ignored and feed the matched-false-alarm detector study without refitting.
