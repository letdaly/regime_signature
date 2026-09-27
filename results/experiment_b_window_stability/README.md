# Experiment B: Fast-kappa window stability

This is a paired 50-replication exploratory sensitivity check of whether raw Signature retains its
class-specific recall advantage for the fast mean-reversion regime (regime 2,
`kappa` in `[14, 18]`) when the causal feature window is changed.

The data-generating process is held fixed: all windows use the same replication
seeds, paths, `p_switch=0.002`, and `min_dwell=50`. This matters because setting
`min_dwell=window_size` would change both the features and the simulated regime
process. Statistics and Signature use the same paths and Random Forest settings
within every replication.

## Leakage status

These recorded runs reuse the primary Experiment B base seed (`20260828`) and
therefore the same test paths on which the fast-class recall pattern was first
noticed. They are valid paired sensitivity calculations, but they are not an
independent confirmation and their confidence intervals and Holm-adjusted
p-values must not be interpreted as confirmatory inference. The runner now
defaults to a fresh seed namespace and refuses overlap with the primary
Experiment B replication seeds.

## Fast-kappa recall result

| Window | Statistics | Signature | Paired delta | 95% CI | Signature wins |
|---:|---:|---:|---:|---:|---:|
| 25 | 0.488 | 0.586 | +0.098 | [+0.070, +0.127] | 88% |
| 50 | 0.530 | 0.661 | +0.131 | [+0.106, +0.156] | 94% |
| 100 | 0.558 | 0.663 | +0.106 | [+0.074, +0.138] | 82% |
| 200 | 0.547 | 0.591 | +0.045 | [+0.013, +0.076] | 66% |

All four paired intervals exclude zero and the descriptive one-sided tests remain below
0.05 after Holm correction across the four windows. Within this exploratory reuse, the **average**
fast-kappa recall advantage is window-stable over `{25, 50, 100, 200}`, although
it weakens substantially and becomes less replication-consistent at window 200.

## Important qualification

This is a class-specific advantage, not a general Signature advantage. Signature
loses slow-kappa recall as the window grows. Its paired balanced-accuracy delta
is +0.009 at window 25, -0.004 at 50, -0.018 at 100, and -0.026 at 200. The
intervals for the last two are entirely below zero. The fast-kappa gain should
therefore be described as a stable representation tradeoff or fast-class bias,
not as evidence that Signature is uniformly the better classifier.

Files:

- `config.json`: frozen design and random seed configuration.
- `replications.csv`: all 200 window-replication results.
- `summary.csv`: exploratory fast-kappa paired intervals and Holm adjustment.
- `tradeoffs.csv`: per-class recall, balanced accuracy, and macro-F1 comparisons.
