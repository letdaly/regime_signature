# Discretization audit of the Design B variance process

Run 2026-09-30 with `audit_discretization.py` (defaults). In Design B all three states share the
stationary variance law Gamma(2 theta / q, q / 2), with mean theta and variance theta q / 2. The
Euler scheme with absorption at zero of `regime_pipeline.HestonPathSimulator` keeps the mean of
the variance process but, while absorption is inactive, inflates its stationary variance by
1 / (1 - kappa delta / 2) for a step delta, so faster states drift further from the matched law.

Protocol: theta = 0.08, q = 0.045 and kappa = 1, 5.25, 16 (midpoints of the Design B ranges);
one and four Euler substeps per daily observation; for every state and step size, 100,000
independent variance paths start from the exact stationary law, the first 2,520 daily
observations are discarded, and moments are averaged over the next 2,520. Standard errors come
from 50 independent batches of paths.

| Substeps | kappa | Variance vs. exact (%) | SE | Euler factor (%) | Mean vs. theta (%) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.00 | +0.26 | 0.20 | +0.20 | +0.06 |
| 1 | 5.25 | +0.92 | 0.11 | +1.05 | -0.02 |
| 1 | 16.00 | +3.22 | 0.06 | +3.28 | -0.01 |
| 4 | 1.00 | -0.12 | 0.21 | +0.05 | -0.10 |
| 4 | 5.25 | +0.02 | 0.10 | +0.26 | -0.07 |
| 4 | 16.00 | +0.84 | 0.05 | +0.80 | +0.02 |

The spread of the variance deviation across the three states is 2.96 percentage points with one
step and 0.96 with four substeps; absorption at zero occurs in fewer than one step in 10,000.
In Design A all states share kappa, so the normalized process V / theta_z follows the same
discretized recursion in every state and the scale-only control is unaffected. Files:
`audit.csv` (all columns) and `config.json`.
