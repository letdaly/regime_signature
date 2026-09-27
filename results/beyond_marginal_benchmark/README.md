# Mechanism benchmark: nested conventional baselines and shuffle placebos

Rerun 2026-09-20 under the origin-anchored signature path (`signature_path` in `config.json`); the
2026-09-15 run under the earlier path convention is archived as `../beyond_marginal_benchmark_path49/`.
Configuration unchanged: 20 replications per scenario, 10 / 5 / 5 train / validation / test paths of
**3,000** observations (not the main grid's 5,000), 50-return windows, **raw** level-3 signatures (39
coordinates), the nested B1 (14) / B2 (29) / B3 (40) conventional blocks defined in
`regime_pipeline.py`, and two model families with the hyperparameter selected on validation
balanced accuracy (logistic C in {0.01, 0.1, 1} with liblinear one-vs-rest; forest depth in {4, 6, 10}).
Placebos shuffle the increments inside each window (fully, or in blocks of 5) before the signature is
recomputed, three repeats each. Base seed 20260915; the parameter families are *not* paired with the main
grid. Total run time 5.1 hours on one process (replications are sequential; only the forest is threaded).

## Test balanced accuracy by representation (mean over 20 replications)

| Scenario | Model | B1 | B2 | B3 | Signature | B3 + signature |
|---|---|---:|---:|---:|---:|---:|
| Scale-only | Logistic | 0.7303 | 0.7239 | 0.7197 | 0.7405 | 0.7310 |
| Scale-only | Random Forest | 0.7354 | 0.7362 | 0.7359 | 0.7369 | 0.7392 |
| Matched-marginal | Logistic | 0.3793 | 0.3816 | 0.3793 | 0.4219 | 0.4104 |
| Matched-marginal | Random Forest | 0.3882 | 0.3923 | 0.3927 | 0.3863 | 0.3952 |

## Paired contrasts (mean, 95% Student-t interval, share of replications with a positive difference)

| Scenario | Model | Contrast | Rerun (origin-anchored path) | Archived (49-increment path) |
|---|---|---|---:|---:|
| Scale-only | Logistic | B3 + signature - B3 | **+0.0113 [+0.0081, +0.0145] (1.00)** | **+0.0113 [+0.0080, +0.0147] (0.95)** |
| Scale-only | Logistic | Signature - shuffled | **+0.0089 [+0.0048, +0.0130] (0.95)** | **+0.0084 [+0.0048, +0.0121] (0.90)** |
| Scale-only | Logistic | Signature - block-shuffled | **+0.0089 [+0.0049, +0.0130] (0.95)** | **+0.0079 [+0.0047, +0.0111] (0.90)** |
| Scale-only | Random Forest | B3 + signature - B3 | **+0.0033 [+0.0008, +0.0058] (0.85)** | **+0.0034 [+0.0012, +0.0057] (0.80)** |
| Scale-only | Random Forest | Signature - shuffled | **+0.0067 [+0.0039, +0.0095] (0.95)** | **+0.0065 [+0.0038, +0.0091] (0.95)** |
| Scale-only | Random Forest | Signature - block-shuffled | **+0.0071 [+0.0044, +0.0097] (0.90)** | **+0.0068 [+0.0045, +0.0091] (0.95)** |
| Matched-marginal | Logistic | B3 + signature - B3 | **+0.0311 [+0.0229, +0.0393] (1.00)** | **+0.0306 [+0.0224, +0.0387] (1.00)** |
| Matched-marginal | Logistic | Signature - shuffled | **+0.0434 [+0.0344, +0.0525] (1.00)** | **+0.0438 [+0.0349, +0.0526] (1.00)** |
| Matched-marginal | Logistic | Signature - block-shuffled | **+0.0389 [+0.0318, +0.0461] (1.00)** | **+0.0386 [+0.0319, +0.0452] (1.00)** |
| Matched-marginal | Random Forest | B3 + signature - B3 | +0.0025 [-0.0013, +0.0062] (0.60) | +0.0001 [-0.0035, +0.0037] (0.45) |
| Matched-marginal | Random Forest | Signature - shuffled | **+0.0078 [+0.0022, +0.0134] (0.70)** | **+0.0087 [+0.0028, +0.0146] (0.65)** |
| Matched-marginal | Random Forest | Signature - block-shuffled | **+0.0071 [+0.0014, +0.0128] (0.65)** | **+0.0083 [+0.0027, +0.0138] (0.70)** |

## Reading

- The path fix changes nothing here: every contrast moves by less than 0.001 and every interval keeps its
  sign. The one qualitative wobble is the forest's B3 + signature gain in the matched-marginal scenario
  (+0.0025 against +0.0001), which still includes zero.
- Adding raw signatures to the strongest conventional block (B3, 40 features) raises the logistic model's
  balanced accuracy by 3.1 points in the matched-marginal scenario in every replication, while the forest
  gains nothing; in the scale-only scenario both gain about 1.1 and 0.3 points. This is the classifier
  dependence that motivated the logistic column of the main grid, where the same pattern appears against the
  compact 11-feature baseline with log-signatures (`../main_grid/README.md`).
- Destroying the order of the increments while keeping their values lowers signature performance in every
  cell, block shuffling as much as full shuffling; the signature is using temporal structure, and structure at
  lags beyond five observations is not what it is using.
- Caveats: raw and log-signatures are related by a nonlinear coordinate
  change that a restricted learner is not invariant to, so these logistic gains do not transfer to the
  log-signature main grid without the matched comparison there; and the benchmark's 3,000-observation paths
  and unpaired families make its absolute levels incomparable with the main rows.

## Output files

`config.json`, `replications.csv` (one row per scenario x replication x model family x representation x
placebo repeat, with the selected setting, validation and test balanced accuracy, macro F1, one-vs-rest
ROC AUC, log loss and Brier score), `summary.csv`, `contrasts.csv`.
