# Market evidence

Both markets run 2026-09-22 with `experiment_market_french.py --market both --model-jobs -1`
(3h13m). The sample was frozen first with `--download --freeze`, which wrote each
`config.json` with the SHA-256 of the archive it read; that commit (`b15a46b`) precedes every
number below.

Daily value-weighted market returns from the Kenneth French Data Library, 1990-07-02 to
2025-12-31 (the frozen end; both archives held data to 2026-07-31). Log returns are
`log(1 + Mkt-RF + RF)`; the portfolio section uses the simple excess return `Mkt-RF`. The
target on day *t* is the tercile of annualized realized volatility over the next 21 trading
days, with cut-points estimated on the training window only (they move little across folds:
0.100-0.107 and 0.153-0.169 annualized in the United States). Features are the 50-return
causal windows and blocks of the unified grid — Statistics (11) extended by HAR averages of
squared returns over 1, 5 and 22 days, B3 (43 here, the same hierarchy plus those HAR
columns), the level-3 raw (39) and log (14) signatures of the origin-anchored three-channel
path, and the forward-filtered probabilities of a three-state Gaussian HMM. The HAR columns
carry the `b1_` prefix, so B3 remains a strict superset of Statistics.

Models are refitted each year on an expanding window, 2005-2025 out of sample. A 21-day
embargo sits before the test year and before the validation block, so no training label
reaches either; the learner (random forest, logistic regression, histogram gradient boosting)
and its setting are chosen on the last three validation years and refitted on the whole
training window. Gains are in percentage points against the corresponding base block with
95% stationary-bootstrap intervals (mean block length 21 days, 2,000 resamples,
reverse-percentile); bold marks intervals excluding zero. Chance balanced accuracy is 33.33%.

## Balanced accuracy, 2005-2025

| Representation | United States | gain [95% CI] | Developed ex US | gain [95% CI] |
|---|---:|---|---:|---|
| HMM filter block | 56.10 | -- | 56.00 | -- |
| Statistics | 54.25 | -- | 54.77 | -- |
| Statistics + log-sig | 56.38 | +2.13 [-0.68, +4.88] | 56.09 | +1.32 [-1.19, +3.84] |
| Statistics + raw-sig | 57.92 | **+3.68 [+1.36, +6.03]** | 54.28 | -0.49 [-3.70, +2.88] |
| B3 | 50.26 | -- | 55.59 | -- |
| B3 + log-sig | 53.87 | **+3.60 [+1.30, +5.98]** | 54.46 | -1.13 [-3.33, +0.99] |
| B3 + raw-sig | 53.98 | **+3.72 [+0.94, +6.31]** | 52.40 | **-3.19 [-5.82, -0.53]** |

5,262 scored days in the United States and 5,457 in developed markets excluding it; the two
calendars differ. Three findings do not match the simulation and belong in the text:

1. The signature gains split by market. Raw signatures add 3.7 points in the United States
   against either baseline, with intervals excluding zero, and add nothing outside it, where
   B3 + raw-sig is resolved in the wrong direction.
2. The Markov-switching filter is the strongest single benchmark here, above Statistics in
   both markets, whereas it falls below Statistics in Design A and sits near chance in
   Design B.
3. B3 is 4.0 points below Statistics in the United States although it contains every
   Statistics column. The learner is selected per representation and B3 chose the random
   forest in 10 of 21 years, so this is not the logistic-only inversion of the simulation
   grid (audited separately in `results/unified_grid_logitc/`).

## Year-by-year stability

The pooled intervals above are considerably tidier than the annual evidence. Per test year,
the paired gain of the validation-selected learner is positive in:

| Contrast | United States | Developed ex US |
|---|---|---|
| Statistics -> + log-sig | 11/21, median +0.74 pp, range [-11.6, +15.8] | 11/21, median +0.25 pp, range [-11.6, +10.0] |
| Statistics -> + raw-sig | 12/21, median +3.19 pp, range [-8.1, +17.8] | 10/21, median +0.00 pp, range [-15.5, +15.7] |
| B3 -> + log-sig | 16/21, median +3.56 pp, range [-4.7, +13.8] | 10/21, median -1.11 pp, range [-19.1, +5.3] |
| B3 -> + raw-sig | 14/21, median +2.74 pp, range [-6.8, +15.4] | 7/21, median -1.76 pp, range [-21.1, +5.3] |

Annual balanced accuracy itself ranges from 24% to 98% in the United States, so single years
are not informative on their own; the stationary bootstrap over the pooled series is the
inferential statement, and the table above is the honest description of its dispersion.

## Volatility-managed exposure

Exposure on day *t+1* is `w_t = c / sigma_hat_t^2` capped at two, where `sigma_hat_t^2`
combines the predicted regime probabilities with the training-window variance of each regime
and `c` equates strategy and market volatility over the training window. Daily excess
returns; mean and volatility annualized in percent; `p` tests equal Sharpe ratios against the
Statistics rule (Ledoit-Wolf, studentized circular-block bootstrap, block 21).

### United States

| Exposure rule | Mean | Vol. | Sharpe | Max DD | Turnover | p |
|---|---:|---:|---:|---:|---:|---:|
| Buy-and-hold | 10.60 | 19.43 | 0.546 | -55.8 | -- | -- |
| HMM filter | 10.17 | 17.35 | 0.586 | -39.9 | 0.117 | 0.78 |
| Statistics | 9.53 | 16.85 | 0.566 | -33.7 | 0.067 | -- |
| Statistics + log-sig | 8.74 | 18.65 | 0.469 | -48.3 | 0.069 | 0.32 |
| B3 + raw-sig | 10.55 | 17.33 | 0.609 | -37.2 | 0.061 | 0.46 |

### Developed ex US

| Exposure rule | Mean | Vol. | Sharpe | Max DD | Turnover | p |
|---|---:|---:|---:|---:|---:|---:|
| Buy-and-hold | 5.80 | 15.73 | 0.369 | -60.2 | -- | -- |
| HMM filter | 6.50 | 15.75 | 0.413 | -48.0 | 0.085 | 0.95 |
| Statistics | 6.23 | 15.21 | 0.410 | -46.6 | 0.055 | -- |
| Statistics + log-sig | 7.29 | 15.45 | 0.472 | -45.1 | 0.060 | 0.14 |
| B3 + raw-sig | 8.42 | 16.02 | 0.525 | -51.2 | 0.081 | 0.10 |

No Sharpe-ratio difference is resolved: every p-value is at least 0.10. Every managed rule
cuts the maximum drawdown, by 22 points in the United States for Statistics, but that is a
property of inverse-variance scaling rather than of the representation. The exercise asks
whether the accuracy differences are economically visible; it is not a proposed strategy.

## Files

`<market>/config.json` is the frozen protocol and the archive hash. `summary.csv` is the
balanced-accuracy table, `economics.csv` the exposure table, `folds.csv` every
year x representation x learner validation and test balanced accuracy with the selected row
flagged, and `cells.csv` how often each learner was selected. `folds/fold_<year>.npz` holds
that year's dates, labels, test probabilities and exposure weights, so Holm corrections,
subperiod splits and new exposure rules need no refitting; the run resumes from them.
