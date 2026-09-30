# Probabilistic evaluation of the market forecasts against conventional benchmarks

Run 2026-09-30 with `market_forecast_evaluation.py` (defaults). The walk-forward folds of the
market study are rebuilt from the frozen data without recomputing any signature; every test date
and label matches the stored fold files exactly. Four benchmarks are fitted on each fold's
training window (training and validation years): climatology (training tercile frequencies),
persistence (tercile of realized volatility over the last 21 days, one-hot), transition
(Laplace-smoothed training frequencies of next-month terciles given the current tercile) and HAR
(log next-month realized volatility on the logs of 1-, 5- and 22-day realized volatility, with the
one-day component floored at its training 1% quantile and Gaussian residuals mapped to tercile
probabilities). The seven stored representations use their validation-selected learners.

Scores are averaged over the 2005-2025 test days: balanced accuracy (BA), the ranked probability
score (RPS, primary for ordered terciles; skill relative to climatology), the logarithmic score
and the Brier score. Contrasts use Diebold-Mariano statistics with a Bartlett HAC variance
(bandwidth 21) for the scores, the stationary bootstrap of the market study for BA (mean block
21, 2,000 resamples), and Holm correction within each family and score.

| Model | US BA (%) | US RPS | US log score | Dev. ex US BA (%) | Dev. ex US RPS | Dev. ex US log score |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 55.81 | 0.2529 | -- | 56.82 | 0.2396 | -- |
| Transition | 55.81 | 0.1753 | 0.957 | 56.82 | 0.1727 | 0.921 |
| HAR | 56.22 | 0.1635 | 0.902 | 57.51 | 0.1604 | 0.867 |
| HMM probability block | 56.10 | 0.1658 | 0.941 | 56.00 | 0.1647 | 0.894 |
| Statistics | 54.25 | 0.1822 | 1.163 | 54.77 | 0.1588 | 0.876 |
| Statistics + log-sig. | 56.38 | 0.1670 | 1.005 | 56.09 | 0.1640 | 0.939 |
| Statistics + raw sig. | 57.92 | 0.1650 | 0.966 | 54.28 | 0.1674 | 1.108 |
| B3 | 50.26 | 0.1818 | 1.041 | 55.59 | 0.1739 | 1.018 |
| B3 + log-sig. | 53.87 | 0.1714 | 0.974 | 54.46 | 0.1703 | 0.957 |
| B3 + raw sig. | 53.98 | 0.1818 | 1.128 | 52.40 | 0.1792 | 1.106 |

Findings:

- In the United States, adding log- or raw signatures to Statistics, and log-signatures to B3,
  lowers the RPS (Holm-adjusted p = 0.023, 0.002 and 0.004); outside the United States no
  augmentation lowers it.
- No representation beats HAR in either market. The best United States forecast by BA,
  Statistics + raw signatures, exceeds HAR by 1.7 points [-1.7, 5.1] (p = 0.30), and its RPS is
  0.0015 higher than HAR's (p = 0.78).

Files: `levels.csv`, `contrasts.csv`, `benchmarks_<market>.npz` (daily benchmark probabilities)
and `config.json`.
