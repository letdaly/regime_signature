# HAR regressions augmented with path signatures

Run 2026-09-30 with `market_har_signatures.py` (defaults). The design was fixed after the
benchmark comparison in `results/market_forecast_evaluation` had shown the log-HAR regression to
be the strongest benchmark, and before any augmented forecast was scored.

For each market and test year, the log-HAR regression of `market_forecast_evaluation.py` is
augmented with standardized extra predictors: the 11 Statistics features (conventional control),
the 14 level-3 log-signature coordinates or the 39 level-3 raw signature coordinates. Only the
extra coefficients are ridge-penalized. The penalty is chosen on the validation years by the mean
ranked probability score (RPS) from {0.1, 1, 10, 100, 1e3, 1e4, 1e5, infinity}; an infinite
penalty returns the plain HAR forecast, which reproduces the HAR benchmark exactly. The model is
refitted on the full training window, and Gaussian residuals give tercile probabilities.
Contrasts with HAR use Diebold-Mariano statistics (Bartlett HAC, bandwidth 21) for the scores,
the stationary bootstrap for balanced accuracy (BA), and Holm correction across the six contrasts
of each score.

| Market | Model | BA (%) | RPS | BA gain vs HAR [95% CI], Holm p | RPS reduction vs HAR [95% CI], Holm p |
|---|---|---:|---:|---|---|
| US | HAR | 56.22 | 0.1635 | -- | -- |
| US | HAR + Statistics | 55.77 | 0.1617 | -0.44 [-2.14, 1.29], 1.00 | 0.0018 [-0.0031, 0.0069], 0.87 |
| US | HAR + log-sig. | 58.49 | 0.1553 | +2.27 [0.39, 4.13], 0.083 | 0.0082 [0.0015, 0.0158], 0.097 |
| US | HAR + raw sig. | 58.92 | 0.1533 | +2.71 [0.54, 4.73], 0.039 | 0.0102 [0.0036, 0.0166], 0.005 |
| Dev. ex US | HAR | 57.51 | 0.1604 | -- | -- |
| Dev. ex US | HAR + Statistics | 56.19 | 0.1592 | -1.32 [-2.92, 0.33], 0.46 | 0.0013 [-0.0032, 0.0060], 0.87 |
| Dev. ex US | HAR + log-sig. | 57.19 | 0.1557 | -0.32 [-1.95, 1.43], 1.00 | 0.0047 [0.0006, 0.0088], 0.097 |
| Dev. ex US | HAR + raw sig. | 56.17 | 0.1572 | -1.34 [-3.69, 1.15], 0.83 | 0.0033 [-0.0030, 0.0095], 0.80 |

In the United States, raw signatures improve the HAR benchmark after Holm correction (BA, RPS
and Brier score), whereas the Statistics features do not; outside the United States no
augmentation is resolved after correction. Files: `levels.csv`, `contrasts.csv`, `penalties.csv`
(selected penalty by year), `forecasts_<market>.npz` (daily probabilities) and `config.json`.
