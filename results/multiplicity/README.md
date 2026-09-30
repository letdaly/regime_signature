# Multiplicity audit

Run 2026-09-23 with `python multiplicity_holm.py`. Nothing is refitted: every contrast is
recomputed from the frozen replication-level outputs of `results/unified_grid`,
`results/unified_grid/detector_matched_fa` and `results/market_french`, and the Holm
step-down correction is applied inside each family and once across all 64 contrasts as a
stress test. `holm.csv` holds every contrast with its raw interval, its raw p-value, and both
adjusted p-values.

| Family | Contrasts | Resolved uncorrected | After Holm in family | After Holm across all 64 |
|---|---:|---:|---:|---:|
| Classification (simulation table) | 24 | 24 | **24** | 24 |
| Detection (four budgets) | 32 | 5 | **0** | 0 |
| Market (accuracy table) | 8 | 4 | **3** | 0 |

Classification: two designs x three learners x four augmentation contrasts, paired t-tests on
the 50 within-replication balanced-accuracy differences behind the Student-t intervals of the
simulation table. Detection: two designs x four false-alarm budgets x four augmentations,
paired t-tests on the detection-probability differences of the validation-selected learner.
Market: two markets x four augmentations, using the stationary-bootstrap p-values of the
walk-forward study, whose resolution is 1/2000.

## What the correction changes

Every classification contrast survives: the largest raw p-value among the 24 is 0.0002
(Design B, random forest, B3 -> + log-sig), which Holm leaves at 0.0002 inside the family and
0.009 across all 64.

No detection contrast survives:

| Design | Budget | Contrast | Gain | Raw p | Holm in family |
|---|---:|---|---:|---:|---:|
| A | 2 | B3 -> + raw-sig | +3.54 pp | 0.0021 | 0.066 |
| A | 5 | Statistics -> + raw-sig | +1.77 pp | 0.0125 | 0.387 |
| B | 2 | Statistics -> + raw-sig | +2.33 pp | 0.0310 | 0.899 |
| B | 5 | Statistics -> + raw-sig | +3.24 pp | 0.0466 | 1.000 |
| B | 5 | B3 -> + raw-sig | +3.17 pp | 0.0272 | 0.815 |

None of the five reaches 5% once the four budgets and four augmentations are corrected
together: no signature augmentation produces a resolved increase in detection probability at
any budget, whereas the classification gains at the same settings are resolved by a wide
margin.

Market: the three United States gains survive within their family (Statistics -> + raw-sig,
p 0.0030 -> 0.021; B3 -> + log-sig, 0.0015 -> 0.012; B3 -> + raw-sig, 0.0075 -> 0.045), and
the negative developed-ex-US contrast does not (B3 -> + raw-sig, 0.0185 -> 0.092), so nothing
is resolved outside the United States in either direction.

## The global column

Pooling all 64 contrasts is a stress test rather than the relevant family, because the three
families answer different questions. Under it the classification family still survives
entirely, and the market gains do not: their smallest p-value (0.0015) ranks 25th and picks
up a multiplier of 40. The market p-values inherit the bootstrap's resolution of 1/2000, so
0.0005 is the smallest value they can take.

The probabilistic scores and HAR comparisons in `results/market_forecast_evaluation` and
`results/market_har_signatures` apply their own Holm corrections, documented in their READMEs.
