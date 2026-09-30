# Volatility-managed exposure driven by the HAR forecasts

Run 2026-09-30 with `market_har_economics.py` (defaults), after the augmented HAR forecasts of
`results/market_har_signatures` had been scored. The exposure rule is the one fixed before the
market analysis: w_t = c / sigma_hat_t^2, capped at two, where sigma_hat_t^2 combines the
predicted tercile probabilities with training-sample tercile variances and c matches the
market's training-sample volatility. Each HAR regression is refitted with its recorded penalty;
its test probabilities reproduce the stored forecasts exactly, and its in-sample probabilities
on the training window set c. Sharpe ratios are compared with the plain HAR rule by the
studentized bootstrap test of the market study (1,000 resamples, block length 21), with Holm
correction across the six comparisons.

| Market | Rule | Sharpe | p vs. HAR | Holm p |
|---|---|---:|---:|---:|
| United States | HAR | 0.598 | -- | -- |
| United States | HAR + Statistics | 0.579 | 0.437 | 1.000 |
| United States | HAR + log-sig. | 0.569 | 0.277 | 1.000 |
| United States | HAR + raw sig. | 0.550 | 0.240 | 1.000 |
| Developed ex US | HAR | 0.339 | -- | -- |
| Developed ex US | HAR + Statistics | 0.324 | 0.594 | 1.000 |
| Developed ex US | HAR + log-sig. | 0.392 | 0.025 | 0.150 |
| Developed ex US | HAR + raw sig. | 0.352 | 0.776 | 1.000 |

The accuracy gains of the augmented HAR forecasts do not translate into resolved Sharpe-ratio
gains. For context, the plain HAR rule does not differ detectably from the Statistics rule of
the market study (p = 0.54 in the United States and 0.10 outside it). Files: `economics.csv`
and `config.json`.
