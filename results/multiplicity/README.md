# Multiplicity audit

Run 2026-09-23 with `python multiplicity_holm.py`. Nothing is refitted: every contrast is
recomputed from the frozen replication-level outputs of `results/unified_grid`,
`results/unified_grid/detector_matched_fa` and `results/market_french`, and the Holm
step-down correction is applied inside each table (the family a reader actually sees) and
once across all 64 contrasts as a stress test. `holm.csv` holds every contrast with its raw
interval, its raw p-value, and both adjusted p-values.

| Family | Contrasts | Resolved uncorrected | After Holm in family | After Holm across all 64 |
|---|---:|---:|---:|---:|
| Classification (Table 1) | 24 | 24 | **24** | 24 |
| Detection (four budgets) | 32 | 5 | **0** | 0 |
| Market (Table 3) | 8 | 4 | **3** | 0 |

Classification: two designs x three learners x four augmentation contrasts, paired t-tests on
the 50 within-replication balanced-accuracy differences behind the Student-t intervals of the
main table. Detection: two designs x four false-alarm budgets x four augmentations, paired
t-tests on the detection-probability differences of the validation-selected learner. Market:
two markets x four augmentations, using the stationary-bootstrap p-values of the walk-forward
study, whose resolution is 1/2000.

## What the correction changes

Every classification contrast survives, and so comfortably that the correction is invisible:
the largest raw p-value among the 24 is 0.0002 (Design B, random forest, B3 -> + log-sig),
which Holm leaves at 0.0002 inside the family and 0.009 across all 64. The claim that
signatures add information to both baselines under all three learners does not depend on
reading one interval at a time.

The detection claims do not survive, and this is the correction that matters:

| Design | Budget | Contrast | Gain | Raw p | Holm in family |
|---|---:|---|---:|---:|---:|
| A | 2 | B3 -> + raw-sig | +3.54 pp | 0.0021 | 0.066 |
| A | 5 | Statistics -> + raw-sig | +1.77 pp | 0.0125 | 0.387 |
| B | 2 | Statistics -> + raw-sig | +2.33 pp | 0.0310 | 0.899 |
| B | 5 | Statistics -> + raw-sig | +3.24 pp | 0.0466 | 1.000 |
| B | 5 | B3 -> + raw-sig | +3.17 pp | 0.0272 | 0.815 |

None of the five reaches 5% once the four budgets and four augmentations are corrected
together. The draft currently says that "at the more permissive target of five false alarms
per 1,000, some raw-signature gains become detectable" and cites three of these numbers; that
sentence is exactly the claim a referee would attack, because it is selected from the budget
at which something happened to clear the line. The corrected statement is simpler and says
the same thing more strongly: no signature augmentation produces a resolved increase in
detection probability at any budget, while the classification gains at the same settings are
resolved by a margin of three orders of magnitude. The gap between recognizing a state and
detecting entry into it is the finding, and multiplicity does not touch it.

Market: the three United States gains survive within their table (Statistics -> + raw-sig,
p 0.0030 -> 0.021; B3 -> + log-sig, 0.0015 -> 0.012; B3 -> + raw-sig, 0.0075 -> 0.045), and
the negative developed-ex-US contrast does not (B3 -> + raw-sig, 0.0185 -> 0.093). The
split-by-market reading therefore holds for the positive side and weakens for the negative
one: outside the United States the honest statement is that nothing is resolved in either
direction.

## The global column

Pooling all 64 contrasts is reported because it is the most hostile reading available, not
because it is the right one: the three families answer different questions, and no referee
asks whether a market gain survives a correction for simulation contrasts. Under it the
classification family still survives entirely, and the market gains do not, their smallest
p-value (0.0015) ranking 25th and picking up a multiplier of 40. Note also that the market
p-values inherit the bootstrap's resolution of 1/2000, so 0.0005 is the smallest value they
can take.

## Reading for the paper

Add one sentence at the end of Section 2.2 saying the intervals are nominal and uncorrected
for multiplicity, with the Holm results in the supplement; replace the "detectable at five
false alarms per 1,000" sentence in Section 3.2 with the corrected statement above; and keep
this table as a supplementary section.
