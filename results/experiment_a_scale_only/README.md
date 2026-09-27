# Experiment A: Scale-Only 50-Replication Study

## Design

- Pure scale contrast: all regimes share `mu`, `kappa`, `rho`, and `c` within
  each independently sampled path.
- Only `theta_i` changes; `sigma_i = c * sqrt(kappa * theta_i)`.
- `theta_0` is drawn from [0.015, 0.025], `theta_1` from [0.065, 0.095], and
  `theta_2` from [0.14, 0.20].
- `V_t / theta_i` follows the same within-regime CIR SDE and has the same
  stationary Gamma law for every regime.
- 50 paired replications; 10/3/5 train/validation/test paths; 5,000 observations
  per path; 50-step causal windows; 200-tree Random Forest.

The equality for normalized variance is exact within fixed-regime dynamics.
At a regime switch, `V_t` remains continuous while the active `theta_i` changes,
so normalized variance jumps and boundary-adjacent windows contain a transient.

## Results

| Metric | Mean | 95% t confidence interval |
|---|---:|---:|
| Balanced accuracy, Statistics | 0.73924 | [0.73234, 0.74615] |
| Balanced accuracy, Signature | 0.74421 | [0.73725, 0.75116] |
| Balanced accuracy, Combined | 0.74664 | [0.73954, 0.75375] |
| Combined minus Statistics | **+0.00740** | **[+0.00600, +0.00880]** |
| Signature minus Statistics | +0.00496 | [+0.00328, +0.00665] |
| Combined minus Signature | +0.00244 | [+0.00169, +0.00318] |

Combined exceeds Statistics in 47 of 50 replications. The paired improvement is
stable but small: about 0.74 percentage points.

## Output files

- `config.json`: frozen scale-only experiment configuration
- `replications.csv`: one row per paired replication
- `summary.csv`: replication-level means and Student-t intervals
- `class_counts.csv`: class counts by split and replication
- `test_diagnostics.csv`: recalls, macro F1, and confusion matrices
