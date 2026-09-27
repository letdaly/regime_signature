# fBM-driven SV: stationary variance audit (2026-09-17)

Long no-switch paths (10,000 observations, stationary initialization) for the
first replication's parameter family of each scenario, under the rough
setting and under the standard-Heston reference (H = 0.5, sigma_scale = 1).
Produced with `experiment_fbm_monte_carlo.stationary_variance_audit`; the
same table is written by `experiment_fbm_monte_carlo.py --audit-only`.
Values below are averaged over the three regimes; `variance_audit.csv` has
every regime.

| Scenario | H | sigma_scale | sd(V) / Heston | mean(V) / theta | Feller 2kt/s^2 | P(V = 0) | return excess kurtosis | acf1(dV) | acf1(abs r) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| matched_marginal | 0.2 | 1 | 0.18 | 0.84 | 2.99 | 0.000 | 0.09 | -0.34 | +0.007 |
| matched_marginal | 0.35 | 1 | 0.40 | 0.88 | 2.99 | 0.000 | 0.24 | -0.20 | +0.028 |
| scale_only | 0.2 | 1 | 0.27 | 0.93 | 7.14 | 0.000 | 0.02 | -0.35 | -0.008 |
| scale_only | 0.2 | 2 | 0.49 | 0.74 | 1.79 | 0.000 | 0.17 | -0.34 | +0.009 |
| scale_only | 0.2 | 3.5 | 0.51 | 0.24 | 0.58 | 0.023 | 1.73 | -0.32 | +0.152 |
| scale_only | 0.35 | 1 | 0.52 | 0.95 | 7.14 | 0.000 | 0.11 | -0.20 | +0.005 |
| matched_marginal (Heston reference) | 0.5 | 1 | 1.00 | ~1 | - | 0 | 1.05 | ~0 | +0.118 |
| scale_only (Heston reference) | 0.5 | 1 | 1.00 | ~1 | - | 0 | 0.50 | ~0 | +0.048 |

## Reading

- With identical parameter families (`sigma_scale = 1`), antipersistent fGn
  (H < 0.5) shrinks the stationary dispersion of the variance to roughly
  20-30% (H = 0.2) or 40-55% (H = 0.35) of the Heston value, pulls mean(V)
  below theta (down to 0.84 theta in the matched-marginal family at H = 0.2,
  where the fast-kappa regime has a large sigma), and removes most of the
  return excess kurtosis. The rough row is therefore *not* marginally matched
  to the Heston row: it is the same parameter family under a different noise
  driver, and in the matched-marginal scenario the three regimes are no
  longer exactly matched to each other either.
- Compensating through `sigma_scale` does not work for the fGn-driven CIR
  recursion: at 2.0 the dispersion ratio only reaches about 0.5 while
  mean(V) already falls below theta; at 3.5 the Feller ratio drops below one,
  the variance is truncated at zero about 2% of the time, mean(V) collapses
  to a quarter of theta, and |returns| acquire spurious lag-one
  autocorrelation of about 0.15.
- Options considered before running the row: accept `sigma_scale = 1`
  and describe the row as 'same Heston families, rough variance driver' with
  the dispersion reduction stated as a limitation, or move to a log-normal
  rough-volatility specification (rough Bergomi-type) whose dispersion is set
  directly by a vol-of-vol parameter without a Feller constraint. The fBM rows
  in `results/main_grid/fbm_*` use the first option.
