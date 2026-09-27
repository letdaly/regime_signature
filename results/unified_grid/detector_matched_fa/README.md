# Regime-change detection at matched false-alarm rates

Run 2026-09-21 with `experiment_matched_false_alarms.py --jobs 12` on the archived probabilities and paths of `results/unified_grid/<design>` (50 replications, 10 validation and 5 test paths of 5,000 observations each). For every detector and every target rate (0.5, 1, 2 and 5 false alarms per 1,000 observations) the operating point is selected on the validation paths (highest detection probability subject to a validation false-alarm rate at or below the target) and applied once to the test paths; the tables report the realized test rate. Classifier detectors pass a cell's archived probabilities through the causal detector of `regime_detection` (smoothing rate in {0.2, 0.5, 1}, confirmations in {1, 3, 5}, 24 thresholds from 0.34 to 0.99). The HMM filter is the grid's three-state Gaussian HMM (five EM restarts, best training likelihood) forward-filtered and passed through the same detector. CUSUM is Page's two-sided cumulative sum on log squared returns relative to an exponentially weighted reference level (allowance in {0.25, 0.5, 1}, reference half-life in {50, 250} observations, 24 geometric decision limits from 2 to 200; sums reset and the reference re-anchors after an alarm). An alarm matches the earliest unmatched true change at most 100 observations before it, whatever state is announced; `state DP` additionally requires the announced state to be the new one. `Chance DP` is the detection probability of a memoryless alarm process firing at the detector's realized total alarm rate, 1 - exp(-(tolerance + 1) x rate), and `DP - chance` its paired difference (pp). All detectors are scored on the same observations (window endpoints from observation 49 on). `selected` rows use, in each replication, the learner with the highest validation balanced accuracy for that representation in the grid. Intervals are 95% Student-t over replications; bold marks paired contrasts whose interval excludes zero.

## Reading

- **Design A.** At the two tight budgets the classical CUSUM is the best detector: 12.4% and 22.8% of changes within 100 observations at 0.5 and 1 false alarm per 1,000, against 5-6% and 17-18% for every classifier-based detector and the HMM filter (paired differences of -7 and -5 pp, all intervals excluding zero). At 2 and 5 per 1,000 the ranking reverses: classifier detectors reach 42-46% and 76-78%, CUSUM 35% and 59%. At 0.5 per 1,000 no classifier detector beats a random alarm process firing at the same rate; from 1 per 1,000 on all do.
- **Signatures do not change detection in Design A.** At 0.5 and 1 per 1,000 every augmentation contrast includes zero for the selected learner, and 22 of the 24 learner-specific contrasts do (the two exceptions are what a 5% level produces by chance); at 2 and 5 per 1,000 raw signatures add 2-4 pp of detection probability for some cells (B3 + raw-sig with the selected learner +3.5 [+1.4, +5.7] at 2 per 1,000; Statistics + raw-sig +1.8 [+0.4, +3.1] at 5 per 1,000). Of 64 delay contrasts per design, 2 exclude zero in each design, again the chance count.
- **Design B.** No detector detects speed changes within 100 observations beyond the chance level by more than a few points: at 1 per 1,000 the best cell (Statistics + raw-sig) reaches 12.4% against a chance level of 11.0%; at 5 per 1,000, 48.0% against 42.6%. CUSUM on log squared returns is at or below chance at every budget (-0.8 pp at 1 and 5 per 1,000), as it must be when all states share one variance law. The classifier detectors nevertheless track the state well once inside a regime (detector balanced accuracy 0.42-0.49 at 5 per 1,000 against 0.33 for the HMM filter), so the states are recognized late rather than not at all, consistent with the regime-age stratification of the main grid.
- **Raw signatures in Design B.** With the selected learner, Statistics + raw-sig adds +2.3 [+0.2, +4.4] pp at 2 per 1,000 and +3.2 [+0.1, +6.4] at 5 per 1,000, and B3 + raw-sig +3.2 [+0.4, +6.0] at 5 per 1,000; log-signatures add nothing resolved. These are an order of magnitude below the classification gains of the same cells.

## Design A (scale-only)

Test paths hold 46.2 true changes per replication on average (24,755 scored observations). Detection probability (DP) within 100 observations and mean delay of matched changes, by target rate.

### Target 0.5 false alarms per 1,000

| Detector | Realized FA / 1,000 | DP | Chance DP | DP - chance | State DP | Mean delay | Detector BA |
|---|---:|---:|---:|---:|---:|---:|---:|
| CUSUM (log squared returns) | 0.48 [0.42, 0.54] | 0.124 [0.108, 0.140] | 0.069 | **+5.54 [+4.30, +6.78]** | n/a | 59.8 [56.9, 62.7] | n/a |
| HMM filter | 0.51 [0.47, 0.55] | 0.049 [0.036, 0.061] | 0.059 | **-1.05 [-1.96, -0.14]** | 0.040 [0.028, 0.053] | 62.8 [55.3, 70.4] | 0.509 [0.490, 0.527] |
| Statistics | 0.46 [0.40, 0.51] | 0.057 [0.045, 0.069] | 0.055 | +0.19 [-0.66, +1.03] | 0.039 [0.029, 0.050] | 55.8 [48.4, 63.2] | 0.507 [0.486, 0.528] |
| Statistics + log-sig | 0.49 [0.45, 0.54] | 0.055 [0.041, 0.070] | 0.058 | -0.32 [-1.43, +0.79] | 0.043 [0.029, 0.057] | 55.2 [47.2, 63.1] | 0.515 [0.496, 0.533] |
| Statistics + raw-sig | 0.50 [0.44, 0.55] | 0.052 [0.040, 0.064] | 0.058 | -0.56 [-1.47, +0.34] | 0.039 [0.027, 0.050] | 56.6 [49.5, 63.6] | 0.505 [0.486, 0.525] |
| B3 | 0.46 [0.40, 0.53] | 0.051 [0.040, 0.063] | 0.055 | -0.34 [-1.15, +0.47] | 0.038 [0.028, 0.047] | 63.3 [56.5, 70.0] | 0.501 [0.480, 0.522] |
| B3 + log-sig | 0.49 [0.44, 0.54] | 0.052 [0.039, 0.066] | 0.057 | -0.50 [-1.60, +0.60] | 0.041 [0.029, 0.054] | 64.2 [57.1, 71.4] | 0.509 [0.491, 0.527] |
| B3 + raw-sig | 0.45 [0.39, 0.51] | 0.052 [0.040, 0.065] | 0.054 | -0.17 [-1.18, +0.84] | 0.037 [0.026, 0.048] | 58.5 [51.4, 65.7] | 0.500 [0.477, 0.522] |

### Target 1.0 false alarms per 1,000

| Detector | Realized FA / 1,000 | DP | Chance DP | DP - chance | State DP | Mean delay | Detector BA |
|---|---:|---:|---:|---:|---:|---:|---:|
| CUSUM (log squared returns) | 0.93 [0.84, 1.03] | 0.228 [0.208, 0.249] | 0.128 | **+10.07 [+8.49, +11.65]** | n/a | 58.3 [55.5, 61.0] | n/a |
| HMM filter | 0.86 [0.79, 0.93] | 0.180 [0.163, 0.196] | 0.113 | **+6.66 [+5.05, +8.28]** | 0.149 [0.133, 0.166] | 65.2 [62.7, 67.6] | 0.591 [0.579, 0.603] |
| Statistics | 0.86 [0.80, 0.93] | 0.182 [0.165, 0.199] | 0.114 | **+6.79 [+5.31, +8.28]** | 0.151 [0.135, 0.167] | 64.7 [61.8, 67.5] | 0.593 [0.581, 0.605] |
| Statistics + log-sig | 0.89 [0.83, 0.95] | 0.182 [0.164, 0.199] | 0.116 | **+6.57 [+5.02, +8.13]** | 0.148 [0.133, 0.162] | 63.6 [61.2, 66.0] | 0.602 [0.589, 0.614] |
| Statistics + raw-sig | 0.87 [0.81, 0.94] | 0.175 [0.157, 0.192] | 0.114 | **+6.09 [+4.55, +7.62]** | 0.139 [0.124, 0.154] | 62.9 [60.7, 65.2] | 0.596 [0.585, 0.607] |
| B3 | 0.85 [0.78, 0.92] | 0.173 [0.156, 0.190] | 0.111 | **+6.18 [+4.72, +7.65]** | 0.138 [0.122, 0.153] | 64.6 [62.1, 67.0] | 0.587 [0.576, 0.599] |
| B3 + log-sig | 0.88 [0.81, 0.95] | 0.183 [0.167, 0.200] | 0.116 | **+6.77 [+5.33, +8.20]** | 0.143 [0.129, 0.157] | 64.6 [62.2, 67.1] | 0.599 [0.589, 0.609] |
| B3 + raw-sig | 0.90 [0.83, 0.96] | 0.175 [0.159, 0.192] | 0.116 | **+5.93 [+4.50, +7.36]** | 0.138 [0.124, 0.151] | 64.9 [62.3, 67.5] | 0.599 [0.587, 0.611] |

### Target 2.0 false alarms per 1,000

| Detector | Realized FA / 1,000 | DP | Chance DP | DP - chance | State DP | Mean delay | Detector BA |
|---|---:|---:|---:|---:|---:|---:|---:|
| CUSUM (log squared returns) | 1.74 [1.61, 1.88] | 0.350 [0.327, 0.373] | 0.214 | **+13.64 [+11.74, +15.55]** | n/a | 56.7 [54.3, 59.1] | n/a |
| HMM filter | 1.98 [1.89, 2.07] | 0.465 [0.438, 0.493] | 0.249 | **+21.59 [+19.17, +24.01]** | 0.282 [0.259, 0.304] | 60.1 [58.4, 61.8] | 0.707 [0.696, 0.719] |
| Statistics | 1.85 [1.75, 1.96] | 0.430 [0.394, 0.465] | 0.234 | **+19.57 [+16.86, +22.28]** | 0.255 [0.232, 0.278] | 59.3 [57.6, 61.1] | 0.691 [0.677, 0.704] |
| Statistics + log-sig | 1.85 [1.71, 1.99] | 0.423 [0.390, 0.456] | 0.232 | **+19.08 [+16.58, +21.58]** | 0.266 [0.244, 0.288] | 59.4 [57.6, 61.2] | 0.694 [0.681, 0.706] |
| Statistics + raw-sig | 1.94 [1.82, 2.07] | 0.445 [0.415, 0.476] | 0.244 | **+20.18 [+17.75, +22.60]** | 0.273 [0.251, 0.294] | 59.4 [57.6, 61.2] | 0.702 [0.691, 0.713] |
| B3 | 1.86 [1.76, 1.95] | 0.426 [0.393, 0.458] | 0.234 | **+19.17 [+16.62, +21.71]** | 0.252 [0.229, 0.275] | 61.5 [59.7, 63.3] | 0.696 [0.684, 0.708] |
| B3 + log-sig | 1.85 [1.75, 1.95] | 0.445 [0.415, 0.474] | 0.236 | **+20.83 [+18.32, +23.34]** | 0.268 [0.244, 0.292] | 60.6 [58.8, 62.4] | 0.702 [0.690, 0.713] |
| B3 + raw-sig | 1.90 [1.80, 2.00] | 0.461 [0.431, 0.492] | 0.242 | **+21.86 [+19.26, +24.45]** | 0.277 [0.255, 0.298] | 60.3 [58.6, 61.9] | 0.705 [0.693, 0.717] |

### Target 5.0 false alarms per 1,000

| Detector | Realized FA / 1,000 | DP | Chance DP | DP - chance | State DP | Mean delay | Detector BA |
|---|---:|---:|---:|---:|---:|---:|---:|
| CUSUM (log squared returns) | 4.64 [4.44, 4.83] | 0.585 [0.563, 0.606] | 0.438 | **+14.73 [+12.84, +16.61]** | n/a | 48.4 [46.8, 50.0] | n/a |
| HMM filter | 4.73 [4.51, 4.95] | 0.748 [0.725, 0.770] | 0.459 | **+28.83 [+26.69, +30.97]** | 0.390 [0.365, 0.415] | 46.1 [44.8, 47.3] | 0.726 [0.716, 0.736] |
| Statistics | 4.71 [4.48, 4.94] | 0.759 [0.741, 0.777] | 0.460 | **+29.91 [+27.78, +32.04]** | 0.386 [0.361, 0.411] | 49.4 [48.2, 50.6] | 0.733 [0.724, 0.741] |
| Statistics + log-sig | 4.64 [4.45, 4.83] | 0.765 [0.746, 0.785] | 0.457 | **+30.83 [+28.70, +32.95]** | 0.406 [0.382, 0.430] | 49.7 [48.6, 50.9] | 0.737 [0.730, 0.745] |
| Statistics + raw-sig | 4.66 [4.43, 4.89] | 0.776 [0.757, 0.796] | 0.459 | **+31.79 [+29.44, +34.13]** | 0.405 [0.382, 0.429] | 48.6 [47.3, 49.9] | 0.744 [0.736, 0.752] |
| B3 | 4.68 [4.46, 4.90] | 0.763 [0.743, 0.784] | 0.459 | **+30.48 [+28.20, +32.76]** | 0.388 [0.363, 0.413] | 50.0 [49.0, 51.0] | 0.735 [0.727, 0.744] |
| B3 + log-sig | 4.66 [4.44, 4.87] | 0.770 [0.749, 0.790] | 0.458 | **+31.17 [+28.94, +33.39]** | 0.390 [0.366, 0.414] | 50.0 [48.9, 51.1] | 0.738 [0.730, 0.746] |
| B3 + raw-sig | 4.59 [4.37, 4.80] | 0.767 [0.745, 0.788] | 0.454 | **+31.28 [+28.81, +33.75]** | 0.403 [0.379, 0.426] | 49.6 [48.5, 50.7] | 0.740 [0.732, 0.748] |

### Detection probability by learner (all targets)

| Cell | Learner | FA 0.5  | FA 1.0  | FA 2.0  | FA 5.0 |
|---|---|---:|---:|---:|---:|
| statistics | RF | 0.054 (FA 0.45) | 0.180 (FA 0.87) | 0.427 (FA 1.88) | 0.760 (FA 4.68) |
| statistics | Logit | 0.057 (FA 0.53) | 0.197 (FA 0.93) | 0.410 (FA 1.91) | 0.749 (FA 4.70) |
| statistics | GBM | 0.041 (FA 0.49) | 0.170 (FA 0.98) | 0.428 (FA 1.92) | 0.758 (FA 4.67) |
| statistics_logsig | RF | 0.049 (FA 0.49) | 0.174 (FA 0.86) | 0.442 (FA 1.86) | 0.764 (FA 4.63) |
| statistics_logsig | Logit | 0.059 (FA 0.51) | 0.209 (FA 0.95) | 0.422 (FA 1.87) | 0.758 (FA 4.68) |
| statistics_logsig | GBM | 0.050 (FA 0.52) | 0.174 (FA 0.99) | 0.451 (FA 1.90) | 0.764 (FA 4.65) |
| statistics_rawsig | RF | 0.054 (FA 0.48) | 0.179 (FA 0.89) | 0.449 (FA 1.93) | 0.769 (FA 4.57) |
| statistics_rawsig | Logit | 0.051 (FA 0.52) | 0.184 (FA 0.93) | 0.444 (FA 1.93) | 0.773 (FA 4.72) |
| statistics_rawsig | GBM | 0.047 (FA 0.50) | 0.179 (FA 1.02) | 0.445 (FA 1.93) | 0.777 (FA 4.72) |
| b3 | RF | 0.052 (FA 0.46) | 0.171 (FA 0.83) | 0.432 (FA 1.86) | 0.763 (FA 4.70) |
| b3 | Logit | 0.045 (FA 0.52) | 0.173 (FA 0.95) | 0.424 (FA 2.01) | 0.743 (FA 4.71) |
| b3 | GBM | 0.045 (FA 0.50) | 0.166 (FA 0.95) | 0.433 (FA 1.94) | 0.758 (FA 4.58) |
| b3_logsig | RF | 0.049 (FA 0.48) | 0.178 (FA 0.85) | 0.449 (FA 1.85) | 0.765 (FA 4.61) |
| b3_logsig | Logit | 0.047 (FA 0.53) | 0.185 (FA 0.96) | 0.425 (FA 1.94) | 0.754 (FA 4.74) |
| b3_logsig | GBM | 0.051 (FA 0.47) | 0.181 (FA 0.99) | 0.449 (FA 1.95) | 0.768 (FA 4.62) |
| b3_rawsig | RF | 0.053 (FA 0.47) | 0.171 (FA 0.87) | 0.470 (FA 1.90) | 0.767 (FA 4.58) |
| b3_rawsig | Logit | 0.053 (FA 0.51) | 0.176 (FA 1.01) | 0.431 (FA 1.93) | 0.761 (FA 4.67) |
| b3_rawsig | GBM | 0.044 (FA 0.48) | 0.182 (FA 1.01) | 0.453 (FA 1.97) | 0.768 (FA 4.73) |

### Paired contrasts at each target (pp of detection probability; delay in observations)

| Contrast | Learner | Target | Delta DP | Delta mean delay | Delta realized FA |
|---|---|---:|---:|---:|---:|
| Statistics -> + log-sig | RF | 1.0 | -0.57 [-1.89, +0.74] 21/50 | -0.7 [-3.0, +1.7] | -0.01 [-0.09, +0.07] |
| Statistics -> + log-sig | RF | 5.0 | +0.45 [-0.65, +1.54] 18/50 | +0.1 [-0.7, +1.0] | -0.05 [-0.21, +0.10] |
| Statistics -> + log-sig | Logit | 1.0 | +1.20 [-0.21, +2.62] 26/50 | +0.4 [-2.5, +3.3] | +0.02 [-0.04, +0.09] |
| Statistics -> + log-sig | Logit | 5.0 | +0.90 [-0.40, +2.21] 23/50 | +0.2 [-1.0, +1.4] | -0.02 [-0.20, +0.16] |
| Statistics -> + log-sig | GBM | 1.0 | +0.45 [-1.39, +2.30] 25/50 | +1.3 [-1.6, +4.3] | +0.01 [-0.04, +0.06] |
| Statistics -> + log-sig | GBM | 5.0 | +0.58 [-0.76, +1.91] 29/50 | -0.4 [-1.3, +0.5] | -0.02 [-0.16, +0.12] |
| Statistics -> + log-sig | selected | 1.0 | -0.04 [-1.61, +1.54] 17/50 | -1.1 [-3.4, +1.2] | +0.02 [-0.05, +0.09] |
| Statistics -> + log-sig | selected | 5.0 | +0.65 [-0.72, +2.02] 19/50 | +0.4 [-0.7, +1.4] | -0.07 [-0.26, +0.11] |
| Statistics -> + raw-sig | RF | 1.0 | -0.05 [-1.35, +1.25] 24/50 | +0.4 [-2.1, +3.0] | +0.02 [-0.05, +0.08] |
| Statistics -> + raw-sig | RF | 5.0 | +0.99 [-0.31, +2.29] 25/50 | -0.5 [-1.3, +0.4] | -0.11 [-0.27, +0.05] |
| Statistics -> + raw-sig | Logit | 1.0 | -1.26 [-3.27, +0.76] 20/50 | -0.9 [-4.0, +2.3] | +0.01 [-0.05, +0.07] |
| Statistics -> + raw-sig | Logit | 5.0 | **+2.36 [+0.79, +3.92]** 30/50 | -0.7 [-2.0, +0.7] | +0.02 [-0.16, +0.19] |
| Statistics -> + raw-sig | GBM | 1.0 | +0.90 [-1.26, +3.05] 26/50 | +2.1 [-1.0, +5.3] | +0.04 [-0.01, +0.09] |
| Statistics -> + raw-sig | GBM | 5.0 | **+1.91 [+0.47, +3.34]** 31/50 | **-1.1 [-2.1, -0.1]** | +0.05 [-0.11, +0.22] |
| Statistics -> + raw-sig | selected | 1.0 | -0.75 [-2.24, +0.74] 23/50 | -1.8 [-4.3, +0.8] | +0.01 [-0.06, +0.07] |
| Statistics -> + raw-sig | selected | 5.0 | **+1.77 [+0.40, +3.14]** 29/50 | -0.8 [-1.6, +0.0] | -0.05 [-0.24, +0.13] |
| B3 -> + log-sig | RF | 1.0 | +0.63 [-0.51, +1.77] 19/50 | +1.2 [-0.6, +3.1] | +0.02 [-0.03, +0.07] |
| B3 -> + log-sig | RF | 5.0 | +0.22 [-0.62, +1.06] 13/50 | +0.1 [-0.7, +0.8] | -0.09 [-0.20, +0.02] |
| B3 -> + log-sig | Logit | 1.0 | +1.13 [-0.19, +2.45] 23/50 | -0.3 [-3.5, +2.9] | +0.01 [-0.05, +0.07] |
| B3 -> + log-sig | Logit | 5.0 | +1.05 [-0.25, +2.36] 25/50 | -1.1 [-2.2, +0.1] | +0.03 [-0.10, +0.15] |
| B3 -> + log-sig | GBM | 1.0 | **+1.49 [+0.02, +2.96]** 26/50 | -1.6 [-4.8, +1.7] | +0.03 [-0.01, +0.08] |
| B3 -> + log-sig | GBM | 5.0 | +1.05 [-0.05, +2.16] 22/50 | **-1.1 [-2.1, -0.2]** | +0.04 [-0.09, +0.18] |
| B3 -> + log-sig | selected | 1.0 | +1.05 [-0.34, +2.44] 20/50 | +0.1 [-2.2, +2.3] | +0.03 [-0.02, +0.09] |
| B3 -> + log-sig | selected | 5.0 | +0.62 [-0.25, +1.49] 17/50 | -0.0 [-0.8, +0.8] | -0.03 [-0.14, +0.09] |
| B3 -> + raw-sig | RF | 1.0 | +0.01 [-1.54, +1.56] 20/50 | +1.2 [-1.3, +3.6] | +0.05 [-0.02, +0.11] |
| B3 -> + raw-sig | RF | 5.0 | +0.46 [-0.56, +1.49] 22/50 | -0.1 [-0.9, +0.7] | -0.12 [-0.25, +0.01] |
| B3 -> + raw-sig | Logit | 1.0 | +0.25 [-1.46, +1.95] 24/50 | -0.8 [-4.0, +2.4] | **+0.06 [+0.00, +0.12]** |
| B3 -> + raw-sig | Logit | 5.0 | **+1.75 [+0.39, +3.10]** 27/50 | -0.7 [-2.1, +0.7] | -0.04 [-0.18, +0.10] |
| B3 -> + raw-sig | GBM | 1.0 | **+1.67 [+0.04, +3.30]** 28/50 | +0.4 [-2.7, +3.4] | **+0.05 [+0.01, +0.10]** |
| B3 -> + raw-sig | GBM | 5.0 | +1.03 [-0.29, +2.35] 21/50 | -0.8 [-2.0, +0.3] | +0.15 [-0.00, +0.31] |
| B3 -> + raw-sig | selected | 1.0 | +0.23 [-1.58, +2.04] 19/50 | +0.3 [-2.4, +3.0] | +0.05 [-0.02, +0.11] |
| B3 -> + raw-sig | selected | 5.0 | +0.31 [-0.83, +1.45] 23/50 | -0.4 [-1.3, +0.5] | -0.10 [-0.26, +0.06] |

### Classifier detectors against the classical baselines (selected learner, pp of DP)

| Representation | Target | minus CUSUM | minus HMM filter |
|---|---:|---:|---:|
| statistics | 1.0 | **-4.63 [-6.81, -2.45]** | +0.23 [-1.60, +2.05] |
| statistics | 5.0 | **+17.39 [+15.23, +19.55]** | +1.10 [-0.56, +2.75] |
| statistics_logsig | 1.0 | **-4.67 [-7.09, -2.25]** | +0.19 [-1.79, +2.16] |
| statistics_logsig | 5.0 | **+18.04 [+15.64, +20.44]** | +1.75 [-0.16, +3.66] |
| statistics_rawsig | 1.0 | **-5.38 [-7.70, -3.06]** | -0.52 [-2.46, +1.41] |
| statistics_rawsig | 5.0 | **+19.16 [+16.60, +21.71]** | **+2.86 [+0.98, +4.74]** |
| b3 | 1.0 | **-5.54 [-8.05, -3.03]** | -0.68 [-2.50, +1.14] |
| b3 | 5.0 | **+17.86 [+15.38, +20.33]** | +1.56 [-0.26, +3.39] |
| b3_logsig | 1.0 | **-4.49 [-6.87, -2.10]** | +0.37 [-1.42, +2.16] |
| b3_logsig | 5.0 | **+18.47 [+15.94, +21.01]** | **+2.18 [+0.33, +4.03]** |
| b3_rawsig | 1.0 | **-5.31 [-7.40, -3.21]** | -0.45 [-2.44, +1.54] |
| b3_rawsig | 5.0 | **+18.17 [+15.50, +20.83]** | **+1.88 [+0.17, +3.58]** |

HMM filter minus CUSUM at 1 / 1,000: **-4.86 [-7.27, -2.44]** pp.

Validation-selected learner shares: statistics: random_forest 84%, logistic 14%, gbm 2%; statistics_logsig: random_forest 52%, logistic 40%, gbm 8%; statistics_rawsig: random_forest 58%, logistic 42%, gbm 0%; b3: random_forest 92%, logistic 4%, gbm 4%; b3_logsig: random_forest 84%, logistic 12%, gbm 4%; b3_rawsig: random_forest 84%, logistic 14%, gbm 2%.

## Design B (matched-marginal)

Test paths hold 46.2 true changes per replication on average (24,755 scored observations). Detection probability (DP) within 100 observations and mean delay of matched changes, by target rate.

### Target 0.5 false alarms per 1,000

| Detector | Realized FA / 1,000 | DP | Chance DP | DP - chance | State DP | Mean delay | Detector BA |
|---|---:|---:|---:|---:|---:|---:|---:|
| CUSUM (log squared returns) | 0.42 [0.35, 0.49] | 0.038 [0.031, 0.045] | 0.048 | **-1.03 [-1.72, -0.35]** | n/a | 48.7 [42.1, 55.3] | n/a |
| HMM filter | 0.46 [0.41, 0.52] | 0.059 [0.048, 0.070] | 0.056 | +0.31 [-0.65, +1.27] | 0.024 [0.018, 0.030] | 50.1 [44.9, 55.3] | 0.340 [0.326, 0.354] |
| Statistics | 0.44 [0.38, 0.50] | 0.062 [0.049, 0.074] | 0.055 | +0.74 [-0.19, +1.67] | 0.028 [0.020, 0.035] | 51.8 [46.8, 56.8] | 0.370 [0.352, 0.388] |
| Statistics + log-sig | 0.45 [0.40, 0.51] | 0.061 [0.051, 0.071] | 0.055 | +0.51 [-0.29, +1.30] | 0.035 [0.028, 0.042] | 49.1 [43.7, 54.6] | 0.385 [0.366, 0.403] |
| Statistics + raw-sig | 0.47 [0.41, 0.52] | 0.062 [0.052, 0.072] | 0.057 | +0.49 [-0.39, +1.36] | 0.039 [0.031, 0.046] | 50.6 [45.5, 55.7] | 0.413 [0.395, 0.430] |
| B3 | 0.44 [0.39, 0.50] | 0.064 [0.052, 0.076] | 0.055 | +0.93 [-0.07, +1.92] | 0.037 [0.029, 0.045] | 49.9 [44.4, 55.4] | 0.370 [0.356, 0.383] |
| B3 + log-sig | 0.47 [0.41, 0.52] | 0.063 [0.052, 0.073] | 0.057 | +0.56 [-0.33, +1.45] | 0.036 [0.028, 0.044] | 51.1 [45.9, 56.4] | 0.371 [0.356, 0.386] |
| B3 + raw-sig | 0.45 [0.39, 0.50] | 0.061 [0.049, 0.073] | 0.055 | +0.64 [-0.40, +1.68] | 0.035 [0.028, 0.042] | 50.8 [45.9, 55.7] | 0.407 [0.387, 0.428] |

### Target 1.0 false alarms per 1,000

| Detector | Realized FA / 1,000 | DP | Chance DP | DP - chance | State DP | Mean delay | Detector BA |
|---|---:|---:|---:|---:|---:|---:|---:|
| CUSUM (log squared returns) | 0.85 [0.74, 0.96] | 0.088 [0.074, 0.102] | 0.096 | -0.81 [-1.74, +0.13] | n/a | 47.3 [43.1, 51.5] | n/a |
| HMM filter | 0.89 [0.80, 0.98] | 0.108 [0.094, 0.123] | 0.104 | +0.43 [-0.70, +1.56] | 0.041 [0.032, 0.049] | 48.7 [44.4, 53.0] | 0.334 [0.320, 0.347] |
| Statistics | 0.91 [0.84, 0.99] | 0.122 [0.107, 0.137] | 0.108 | **+1.37 [+0.21, +2.53]** | 0.059 [0.048, 0.069] | 49.3 [45.6, 53.0] | 0.386 [0.370, 0.402] |
| Statistics + log-sig | 0.87 [0.80, 0.94] | 0.111 [0.099, 0.123] | 0.103 | +0.78 [-0.27, +1.82] | 0.060 [0.051, 0.069] | 51.6 [47.4, 55.8] | 0.410 [0.394, 0.426] |
| Statistics + raw-sig | 0.92 [0.85, 1.00] | 0.124 [0.108, 0.141] | 0.110 | **+1.46 [+0.09, +2.84]** | 0.073 [0.060, 0.086] | 50.6 [47.1, 54.0] | 0.448 [0.434, 0.461] |
| B3 | 0.92 [0.85, 0.99] | 0.114 [0.099, 0.129] | 0.108 | +0.59 [-0.68, +1.86] | 0.063 [0.051, 0.074] | 45.4 [41.3, 49.6] | 0.402 [0.386, 0.417] |
| B3 + log-sig | 0.88 [0.80, 0.96] | 0.117 [0.102, 0.131] | 0.104 | +1.21 [-0.07, +2.49] | 0.060 [0.050, 0.070] | 49.2 [45.4, 53.0] | 0.405 [0.392, 0.418] |
| B3 + raw-sig | 0.90 [0.81, 0.99] | 0.120 [0.102, 0.138] | 0.107 | +1.35 [-0.20, +2.89] | 0.071 [0.059, 0.083] | 50.4 [46.2, 54.6] | 0.441 [0.428, 0.455] |

### Target 2.0 false alarms per 1,000

| Detector | Realized FA / 1,000 | DP | Chance DP | DP - chance | State DP | Mean delay | Detector BA |
|---|---:|---:|---:|---:|---:|---:|---:|
| CUSUM (log squared returns) | 1.89 [1.73, 2.05] | 0.182 [0.159, 0.204] | 0.200 | -1.79 [-3.71, +0.13] | n/a | 44.9 [41.7, 48.1] | n/a |
| HMM filter | 1.81 [1.68, 1.94] | 0.205 [0.186, 0.224] | 0.197 | +0.73 [-0.78, +2.23] | 0.066 [0.056, 0.077] | 48.0 [44.4, 51.6] | 0.333 [0.320, 0.346] |
| Statistics | 1.81 [1.70, 1.91] | 0.225 [0.205, 0.245] | 0.201 | **+2.42 [+0.71, +4.13]** | 0.107 [0.091, 0.124] | 48.0 [45.4, 50.6] | 0.404 [0.390, 0.418] |
| Statistics + log-sig | 1.88 [1.78, 1.98] | 0.228 [0.208, 0.248] | 0.207 | **+2.09 [+0.41, +3.76]** | 0.113 [0.098, 0.128] | 48.5 [45.9, 51.0] | 0.437 [0.424, 0.449] |
| Statistics + raw-sig | 1.86 [1.74, 1.98] | 0.248 [0.230, 0.267] | 0.208 | **+4.05 [+2.35, +5.74]** | 0.136 [0.122, 0.151] | 49.3 [46.6, 52.0] | 0.474 [0.464, 0.484] |
| B3 | 1.87 [1.73, 2.01] | 0.215 [0.197, 0.233] | 0.204 | +1.14 [-0.71, +2.99] | 0.107 [0.093, 0.120] | 48.3 [45.8, 50.8] | 0.422 [0.409, 0.435] |
| B3 + log-sig | 1.88 [1.76, 2.01] | 0.232 [0.208, 0.255] | 0.208 | **+2.40 [+0.34, +4.47]** | 0.116 [0.099, 0.132] | 46.5 [43.9, 49.1] | 0.429 [0.415, 0.442] |
| B3 + raw-sig | 1.82 [1.71, 1.93] | 0.235 [0.214, 0.256] | 0.203 | **+3.20 [+1.53, +4.87]** | 0.129 [0.115, 0.143] | 52.0 [48.9, 55.2] | 0.464 [0.451, 0.477] |

### Target 5.0 false alarms per 1,000

| Detector | Realized FA / 1,000 | DP | Chance DP | DP - chance | State DP | Mean delay | Detector BA |
|---|---:|---:|---:|---:|---:|---:|---:|
| CUSUM (log squared returns) | 4.47 [4.19, 4.75] | 0.398 [0.374, 0.423] | 0.406 | -0.74 [-2.60, +1.12] | n/a | 42.8 [40.7, 45.0] | n/a |
| HMM filter | 4.66 [4.48, 4.84] | 0.470 [0.452, 0.489] | 0.427 | **+4.31 [+2.66, +5.95]** | 0.153 [0.137, 0.168] | 45.6 [43.7, 47.5] | 0.331 [0.319, 0.344] |
| Statistics | 4.57 [4.32, 4.83] | 0.448 [0.423, 0.473] | 0.419 | **+2.92 [+0.80, +5.04]** | 0.203 [0.187, 0.219] | 44.8 [42.6, 47.0] | 0.419 [0.408, 0.430] |
| Statistics + log-sig | 4.69 [4.49, 4.89] | 0.453 [0.428, 0.479] | 0.427 | **+2.64 [+0.47, +4.82]** | 0.214 [0.197, 0.231] | 46.4 [44.6, 48.1] | 0.447 [0.437, 0.457] |
| Statistics + raw-sig | 4.64 [4.45, 4.82] | 0.480 [0.452, 0.509] | 0.426 | **+5.39 [+2.88, +7.90]** | 0.248 [0.227, 0.269] | 45.5 [43.7, 47.4] | 0.486 [0.477, 0.496] |
| B3 | 4.71 [4.51, 4.91] | 0.440 [0.417, 0.463] | 0.427 | +1.38 [-0.61, +3.37] | 0.199 [0.183, 0.215] | 45.1 [43.3, 46.9] | 0.431 [0.417, 0.445] |
| B3 + log-sig | 4.56 [4.37, 4.75] | 0.437 [0.414, 0.460] | 0.417 | +1.94 [-0.05, +3.94] | 0.202 [0.184, 0.220] | 44.6 [42.8, 46.4] | 0.441 [0.428, 0.454] |
| B3 + raw-sig | 4.65 [4.45, 4.84] | 0.472 [0.448, 0.496] | 0.426 | **+4.61 [+2.64, +6.59]** | 0.227 [0.210, 0.244] | 45.4 [43.7, 47.2] | 0.475 [0.465, 0.485] |

### Detection probability by learner (all targets)

| Cell | Learner | FA 0.5  | FA 1.0  | FA 2.0  | FA 5.0 |
|---|---|---:|---:|---:|---:|
| statistics | RF | 0.056 (FA 0.46) | 0.117 (FA 0.93) | 0.213 (FA 1.91) | 0.406 (FA 4.52) |
| statistics | Logit | 0.058 (FA 0.45) | 0.116 (FA 0.90) | 0.226 (FA 1.79) | 0.465 (FA 4.61) |
| statistics | GBM | 0.066 (FA 0.47) | 0.111 (FA 0.91) | 0.212 (FA 1.87) | 0.437 (FA 4.66) |
| statistics_logsig | RF | 0.053 (FA 0.43) | 0.110 (FA 0.91) | 0.219 (FA 1.82) | 0.423 (FA 4.36) |
| statistics_logsig | Logit | 0.062 (FA 0.47) | 0.112 (FA 0.87) | 0.227 (FA 1.90) | 0.446 (FA 4.68) |
| statistics_logsig | GBM | 0.055 (FA 0.47) | 0.120 (FA 0.91) | 0.234 (FA 1.88) | 0.452 (FA 4.64) |
| statistics_rawsig | RF | 0.050 (FA 0.42) | 0.109 (FA 0.92) | 0.197 (FA 1.94) | 0.413 (FA 4.61) |
| statistics_rawsig | Logit | 0.062 (FA 0.47) | 0.124 (FA 0.92) | 0.248 (FA 1.86) | 0.480 (FA 4.64) |
| statistics_rawsig | GBM | 0.063 (FA 0.45) | 0.121 (FA 0.97) | 0.226 (FA 1.85) | 0.440 (FA 4.66) |
| b3 | RF | 0.063 (FA 0.43) | 0.113 (FA 0.90) | 0.209 (FA 1.83) | 0.414 (FA 4.56) |
| b3 | Logit | 0.052 (FA 0.40) | 0.105 (FA 0.89) | 0.214 (FA 1.86) | 0.468 (FA 4.72) |
| b3 | GBM | 0.057 (FA 0.46) | 0.113 (FA 0.91) | 0.212 (FA 1.84) | 0.446 (FA 4.64) |
| b3_logsig | RF | 0.057 (FA 0.43) | 0.110 (FA 0.89) | 0.219 (FA 1.78) | 0.401 (FA 4.48) |
| b3_logsig | Logit | 0.061 (FA 0.46) | 0.110 (FA 0.88) | 0.224 (FA 1.91) | 0.466 (FA 4.68) |
| b3_logsig | GBM | 0.064 (FA 0.45) | 0.121 (FA 0.95) | 0.228 (FA 1.90) | 0.464 (FA 4.71) |
| b3_rawsig | RF | 0.056 (FA 0.43) | 0.105 (FA 0.87) | 0.209 (FA 1.84) | 0.424 (FA 4.54) |
| b3_rawsig | Logit | 0.061 (FA 0.45) | 0.120 (FA 0.90) | 0.235 (FA 1.82) | 0.472 (FA 4.65) |
| b3_rawsig | GBM | 0.058 (FA 0.46) | 0.113 (FA 0.95) | 0.229 (FA 1.90) | 0.467 (FA 4.72) |

### Paired contrasts at each target (pp of detection probability; delay in observations)

| Contrast | Learner | Target | Delta DP | Delta mean delay | Delta realized FA |
|---|---|---:|---:|---:|---:|
| Statistics -> + log-sig | RF | 1.0 | -0.71 [-2.21, +0.79] 17/50 | -0.5 [-4.2, +3.2] | -0.02 [-0.10, +0.06] |
| Statistics -> + log-sig | RF | 5.0 | +1.67 [-1.26, +4.60] 28/50 | +0.1 [-1.9, +2.1] | -0.16 [-0.41, +0.09] |
| Statistics -> + log-sig | Logit | 1.0 | -0.36 [-1.82, +1.11] 17/50 | +0.5 [-5.0, +5.9] | -0.03 [-0.12, +0.05] |
| Statistics -> + log-sig | Logit | 5.0 | -1.87 [-4.39, +0.65] 20/50 | -1.1 [-3.8, +1.6] | +0.08 [-0.13, +0.29] |
| Statistics -> + log-sig | GBM | 1.0 | +0.86 [-0.58, +2.30] 23/50 | -1.9 [-7.2, +3.4] | -0.00 [-0.10, +0.10] |
| Statistics -> + log-sig | GBM | 5.0 | +1.44 [-0.86, +3.73] 27/50 | +1.3 [-0.5, +3.1] | -0.01 [-0.18, +0.16] |
| Statistics -> + log-sig | selected | 1.0 | -1.13 [-2.53, +0.27] 13/50 | +2.2 [-2.9, +7.4] | -0.04 [-0.13, +0.05] |
| Statistics -> + log-sig | selected | 5.0 | +0.53 [-2.44, +3.49] 26/50 | +1.5 [-0.7, +3.8] | +0.12 [-0.14, +0.37] |
| Statistics -> + raw-sig | RF | 1.0 | -0.77 [-2.27, +0.73] 18/50 | +0.9 [-3.3, +5.1] | -0.00 [-0.09, +0.09] |
| Statistics -> + raw-sig | RF | 5.0 | +0.69 [-2.37, +3.74] 24/50 | +0.5 [-1.4, +2.4] | +0.09 [-0.15, +0.32] |
| Statistics -> + raw-sig | Logit | 1.0 | +0.88 [-0.92, +2.67] 25/50 | -0.8 [-6.2, +4.6] | +0.03 [-0.06, +0.11] |
| Statistics -> + raw-sig | Logit | 5.0 | +1.52 [-1.44, +4.48] 29/50 | -1.1 [-3.5, +1.4] | +0.03 [-0.20, +0.26] |
| Statistics -> + raw-sig | GBM | 1.0 | +0.94 [-0.52, +2.39] 24/50 | -0.1 [-5.3, +5.2] | +0.05 [-0.03, +0.14] |
| Statistics -> + raw-sig | GBM | 5.0 | +0.30 [-2.07, +2.68] 19/50 | +0.2 [-1.6, +2.1] | +0.00 [-0.20, +0.20] |
| Statistics -> + raw-sig | selected | 1.0 | +0.25 [-1.48, +1.97] 16/50 | +1.2 [-3.3, +5.7] | +0.01 [-0.08, +0.10] |
| Statistics -> + raw-sig | selected | 5.0 | **+3.24 [+0.05, +6.43]** 28/50 | +0.7 [-1.6, +3.0] | +0.06 [-0.21, +0.33] |
| B3 -> + log-sig | RF | 1.0 | -0.37 [-1.43, +0.69] 15/50 | +2.6 [-1.3, +6.4] | -0.01 [-0.08, +0.06] |
| B3 -> + log-sig | RF | 5.0 | -1.34 [-3.40, +0.72] 15/50 | -0.1 [-1.4, +1.1] | -0.08 [-0.25, +0.09] |
| B3 -> + log-sig | Logit | 1.0 | +0.49 [-1.24, +2.22] 22/50 | -1.4 [-5.6, +2.8] | -0.01 [-0.10, +0.08] |
| B3 -> + log-sig | Logit | 5.0 | -0.29 [-2.30, +1.72] 23/50 | -0.3 [-2.8, +2.1] | -0.04 [-0.20, +0.11] |
| B3 -> + log-sig | GBM | 1.0 | +0.85 [-0.58, +2.27] 26/50 | +1.3 [-1.8, +4.3] | +0.04 [-0.04, +0.12] |
| B3 -> + log-sig | GBM | 5.0 | +1.81 [-0.45, +4.06] 27/50 | -0.3 [-1.8, +1.1] | +0.07 [-0.12, +0.27] |
| B3 -> + log-sig | selected | 1.0 | +0.26 [-1.10, +1.62] 19/50 | +3.7 [-1.0, +8.5] | -0.04 [-0.14, +0.05] |
| B3 -> + log-sig | selected | 5.0 | -0.36 [-2.52, +1.81] 24/50 | -0.5 [-2.0, +1.0] | -0.15 [-0.36, +0.06] |
| B3 -> + raw-sig | RF | 1.0 | -0.87 [-2.20, +0.47] 12/50 | +4.9 [-1.3, +11.2] | -0.03 [-0.10, +0.04] |
| B3 -> + raw-sig | RF | 5.0 | +0.96 [-1.26, +3.18] 27/50 | -0.4 [-1.8, +1.0] | -0.02 [-0.20, +0.17] |
| B3 -> + raw-sig | Logit | 1.0 | +1.49 [-0.34, +3.31] 26/50 | -1.3 [-7.1, +4.5] | +0.01 [-0.10, +0.11] |
| B3 -> + raw-sig | Logit | 5.0 | +0.36 [-2.22, +2.95] 24/50 | +0.9 [-1.0, +2.9] | -0.08 [-0.31, +0.15] |
| B3 -> + raw-sig | GBM | 1.0 | +0.07 [-1.39, +1.52] 23/50 | +2.0 [-2.5, +6.5] | +0.04 [-0.03, +0.11] |
| B3 -> + raw-sig | GBM | 5.0 | **+2.15 [+0.05, +4.24]** 28/50 | +0.2 [-1.4, +1.9] | +0.08 [-0.09, +0.25] |
| B3 -> + raw-sig | selected | 1.0 | +0.61 [-1.38, +2.61] 23/50 | +4.3 [-1.3, +9.8] | -0.02 [-0.12, +0.08] |
| B3 -> + raw-sig | selected | 5.0 | **+3.17 [+0.37, +5.97]** 26/50 | +0.3 [-1.5, +2.1] | -0.06 [-0.30, +0.18] |

### Classifier detectors against the classical baselines (selected learner, pp of DP)

| Representation | Target | minus CUSUM | minus HMM filter |
|---|---:|---:|---:|
| statistics | 1.0 | **+3.39 [+1.59, +5.19]** | +1.38 [-0.53, +3.29] |
| statistics | 5.0 | **+4.97 [+2.05, +7.89]** | -2.25 [-5.10, +0.59] |
| statistics_logsig | 1.0 | **+2.26 [+0.67, +3.86]** | +0.25 [-1.61, +2.11] |
| statistics_logsig | 5.0 | **+5.50 [+2.37, +8.62]** | -1.73 [-4.31, +0.86] |
| statistics_rawsig | 1.0 | **+3.64 [+1.74, +5.54]** | +1.63 [-0.71, +3.97] |
| statistics_rawsig | 5.0 | **+8.21 [+5.15, +11.27]** | +0.99 [-1.89, +3.86] |
| b3 | 1.0 | **+2.60 [+0.66, +4.54]** | +0.59 [-1.07, +2.25] |
| b3 | 5.0 | **+4.21 [+1.45, +6.97]** | **-3.01 [-5.34, -0.68]** |
| b3_logsig | 1.0 | **+2.86 [+0.97, +4.75]** | +0.85 [-1.01, +2.71] |
| b3_logsig | 5.0 | **+3.86 [+1.14, +6.57]** | **-3.37 [-5.96, -0.77]** |
| b3_rawsig | 1.0 | **+3.21 [+1.21, +5.22]** | +1.20 [-1.20, +3.61] |
| b3_rawsig | 5.0 | **+7.38 [+4.68, +10.09]** | +0.16 [-2.71, +3.03] |

HMM filter minus CUSUM at 1 / 1,000: **+2.01 [+0.06, +3.96]** pp.

Validation-selected learner shares: statistics: random_forest 44%, logistic 40%, gbm 16%; statistics_logsig: random_forest 10%, logistic 72%, gbm 18%; statistics_rawsig: random_forest 0%, logistic 100%, gbm 0%; b3: random_forest 78%, logistic 8%, gbm 14%; b3_logsig: random_forest 48%, logistic 50%, gbm 2%; b3_rawsig: random_forest 0%, logistic 100%, gbm 0%.

## Sensitivity: tolerance 250 observations

Same calibration with matches allowed up to 250 observations after a change (`--tolerance 250`, `detector_matched_fa_tol250/`). Detection probability at 1 and 5 false alarms per 1,000 with the chance level in parentheses.

| Detector | A: FA 1 | A: FA 5 | B: FA 1 | B: FA 5 |
|---|---:|---:|---:|---:|
| CUSUM (log squared returns) | 0.587 (0.406) | 0.835 (0.744) | 0.173 (0.242) | 0.680 (0.766) |
| HMM filter | 0.734 (0.442) | 0.930 (0.778) | 0.278 (0.300) | 0.775 (0.778) |
| Statistics | 0.620 (0.382) | 0.934 (0.760) | 0.297 (0.316) | 0.739 (0.773) |
| Statistics + log-sig | 0.681 (0.409) | 0.938 (0.752) | 0.296 (0.307) | 0.742 (0.776) |
| Statistics + raw-sig | 0.665 (0.407) | 0.942 (0.763) | 0.337 (0.317) | 0.773 (0.774) |
| B3 | 0.648 (0.391) | 0.936 (0.752) | 0.291 (0.311) | 0.723 (0.771) |
| B3 + log-sig | 0.687 (0.415) | 0.937 (0.749) | 0.292 (0.311) | 0.744 (0.774) |
| B3 + raw-sig | 0.691 (0.415) | 0.940 (0.764) | 0.323 (0.315) | 0.776 (0.772) |

## Output files

`figure_tradeoff.png` (detection probability against realized false-alarm rate, both designs); per design `config.json`, `replications.csv` (one row per replication x detector x learner x target with the selected setting, validation and test metrics; `learner = selected` rows duplicate the validation-selected learner), `summary.csv` (means and intervals), `contrasts.csv` (paired differences: augmented minus base within learner, representation minus CUSUM / HMM filter, HMM filter minus CUSUM).
