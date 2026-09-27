# Logistic penalty audit: does B3 stay below Statistics?

Both designs run 2026-09-22/23 with

```
experiment_unified_grid.py --design <design> --learners logistic \
  --cells statistics statistics_logsig statistics_rawsig b3 b3_logsig b3_rawsig \
  --logistic-c-grid 0.0001 0.001 0.01 0.03 0.1 0.3 1.0 10.0 100.0 \
  --no-save-probabilities
```

Everything except the penalty search matches `results/unified_grid/`: the same base seeds, so
replication *k* simulates the same paths, the same 50-return windows, the same blocks, the
same validation-only selection rule. The frozen grid searches three inverse penalties
(0.01 / 0.1 / 1); this run searches nine spanning 1e-4 to 1e2. The comparison below is
therefore paired within replication, and the "9 minus 3" column isolates the tuning effect.
The Markov-switching block is not fitted here (no cell needs it) and no probabilities are
archived. Gains are in percentage points with 95% Student-t intervals over 50 replications.

## Effect of the wider search

| Cell | Design A: 3-point | 9-point | 9 minus 3 | Design B: 3-point | 9-point | 9 minus 3 |
|---|---:|---:|---|---:|---:|---|
| Statistics | 73.03 | 73.01 | -0.01 [-0.04, +0.01] | 39.78 | 39.65 | -0.13 [-0.31, +0.05] |
| Statistics + log-sig | 73.85 | 73.84 | -0.02 [-0.06, +0.02] | 41.66 | 41.62 | -0.04 [-0.14, +0.06] |
| Statistics + raw-sig | 74.33 | 74.35 | +0.02 [-0.02, +0.06] | 44.31 | 44.32 | +0.01 [-0.03, +0.05] |
| B3 | 72.61 | 72.57 | -0.04 [-0.15, +0.06] | 39.29 | 39.16 | -0.13 [-0.30, +0.05] |
| B3 + log-sig | 73.10 | 73.09 | -0.01 [-0.12, +0.10] | 40.74 | 40.74 | -0.00 [-0.05, +0.04] |
| B3 + raw-sig | 73.65 | 73.70 | +0.06 [-0.06, +0.17] | 43.35 | 43.30 | -0.05 [-0.08, -0.01] |

Tripling the penalty search moves no cell by more than 0.13 points, and every interval but
one covers zero; the exception, B3 + raw-sig in Design B, is -0.05 points, in the wrong
direction for the tuning explanation. The paper's logistic column is not a tuning artefact.

## The inversion itself

Paired differences, superset minus subset. B3 contains every Statistics column, so a
well-tuned linear model should not be worse.

| Contrast | Design A: 3-point | 9-point | Design B: 3-point | 9-point |
|---|---|---|---|---|
| B3 - Statistics | -0.42 [-0.60, -0.24] 15/50 | -0.45 [-0.64, -0.25] 13/50 | -0.49 [-0.91, -0.06] 17/50 | -0.48 [-0.88, -0.09] 18/50 |
| B3 + log-sig - Statistics + log-sig | -0.75 [-0.92, -0.59] 6/50 | -0.74 [-0.92, -0.57] 5/50 | -0.92 [-1.22, -0.61] 10/50 | -0.88 [-1.21, -0.55] 11/50 |
| B3 + raw-sig - Statistics + raw-sig | -0.68 [-0.83, -0.54] 4/50 | -0.65 [-0.81, -0.49] 7/50 | -0.96 [-1.30, -0.62] 12/50 | -1.02 [-1.35, -0.68] 10/50 |

The inversion survives the wider search unchanged in both designs and in all three pairs.

## Which penalty wins

| Cell | Design A: median C | at 1e-4 | at 1e2 | Design B: median C | at 1e-4 | at 1e2 |
|---|---:|---:|---:|---:|---:|---:|
| Statistics | 0.1 | 0/50 | 11/50 | 0.1 | 4/50 | 5/50 |
| B3 | 0.1 | 1/50 | 6/50 | 0.3 | 8/50 | 13/50 |
| Statistics + raw-sig | 5.5 | 0/50 | 14/50 | 1 | 0/50 | 10/50 |
| B3 + raw-sig | 1 | 1/50 | 11/50 | 1 | 0/50 | 16/50 |

Two things follow. The selection lands at the weak-penalty endpoint (C = 100, effectively
unregularized) in 6 to 16 replications of every cell, so B3 is not being shrunk out of its
advantage; given the freedom to use its 29 extra columns, validation balanced accuracy still
prefers not to. And the selected value scatters across the whole range, including both
endpoints of the same cell in different replications, which says the validation surface is
nearly flat in C: the penalty is not what separates these blocks.

## Standardization is not the cause either

The logistic pipeline scales features robustly (10th-90th percentile, unit variance) and
clips the result at +/-20 to keep the optimization conditioned, so the other candidate
explanation is that the clip removes what the extra columns carry. It does not. On
replication 1 of Design A, over its 24,755 test windows, no Statistics value is clipped
(0.0000%) and 0.0035% of the 29 extra B3 values are, the worst single column being realized
quarticity at 0.10%; no extra column is degenerate in the training window. Whatever the extra
columns contribute reaches the model intact.

## Reading for the paper

The B3-below-Statistics inversion is a property of the feature set at this sample size, not
of the hyperparameter grid: the 29 conventional features beyond Statistics are net noise for
a linear model, while the trees, which can ignore a column outright, gain from them. The
frozen three-point grid stays in the main text, where it keeps the search budgets of the
three learners equal at three settings each; this audit belongs in the supplement as the
robustness check behind the one sentence that concedes the inversion.
