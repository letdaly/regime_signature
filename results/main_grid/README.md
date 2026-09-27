# Main grid: 3 designs x 3 representations x 2 learners

All rows: 50 paired replications, 10 train / 10 validation / 5 test paths of 5,000 observations, 50-return causal
windows labelled by their endpoint state, level-3 log-signatures on the origin-anchored three-channel path
(Statistics 11 / Signature 14 / Combined 25 columns of one feature matrix), a fixed Random Forest and a
validation-tuned logistic regression on identical column subsets, validation-tuned causal detector, 95% Student-t
intervals over replications. The fBM rows reuse the paired Heston row's base seed, so replication k is the same
parameter family under rough variance noise (`sigma_scale = 1`; the marginal mismatch is documented in each
row's audit). Heston rows rerun 2026-09-19 (archived 2026-09-18 runs in `*_path49/`), fBM rows run 2026-09-20.
Bold marks intervals that exclude zero. Every number below is read from the row's `summary.csv`.

## Test balanced accuracy, Random Forest

| Row | Statistics | Signature | Combined | Combined - Statistics [95% CI] | Positive | Signature - Statistics |
|---|---:|---:|---:|---:|---:|---:|
| Heston, scale-only | 0.7353 | 0.7306 | 0.7382 | **+0.0029 [+0.0022, +0.0036]** | 44/50 | -0.0047 |
| fBM-driven, scale-only, H = 0.2 | 0.8024 | 0.7960 | 0.8035 | **+0.0011 [+0.0008, +0.0014]** | 39/50 | -0.0063 |
| fBM-driven, scale-only, H = 0.35 | 0.7874 | 0.7808 | 0.7887 | **+0.0013 [+0.0008, +0.0019]** | 39/50 | -0.0065 |
| Heston, matched-marginal | 0.3952 | 0.3998 | 0.4027 | **+0.0075 [+0.0053, +0.0097]** | 43/50 | +0.0047 |
| fBM-driven, matched-marginal, H = 0.2 | 0.3430 | 0.3376 | 0.3456 | **+0.0025 [+0.0012, +0.0038]** | 39/50 | -0.0055 |
| fBM-driven, matched-marginal, H = 0.35 | 0.3583 | 0.3582 | 0.3613 | **+0.0030 [+0.0014, +0.0046]** | 33/50 | -0.0000 |

## Test balanced accuracy, logistic regression

| Row | Statistics | Signature | Combined | Combined - Statistics [95% CI] | Positive | Signature - Statistics |
|---|---:|---:|---:|---:|---:|---:|
| Heston, scale-only | 0.7303 | 0.7355 | 0.7385 | **+0.0083 [+0.0068, +0.0097]** | 46/50 | +0.0052 |
| fBM-driven, scale-only, H = 0.2 | 0.7955 | 0.7904 | 0.7973 | **+0.0018 [+0.0007, +0.0028]** | 34/50 | -0.0051 |
| fBM-driven, scale-only, H = 0.35 | 0.7811 | 0.7781 | 0.7847 | **+0.0035 [+0.0025, +0.0046]** | 43/50 | -0.0030 |
| Heston, matched-marginal | 0.3978 | 0.3953 | 0.4166 | **+0.0188 [+0.0159, +0.0217]** | 49/50 | -0.0025 |
| fBM-driven, matched-marginal, H = 0.2 | 0.3421 | 0.3437 | 0.3475 | **+0.0054 [+0.0026, +0.0082]** | 33/50 | +0.0016 |
| fBM-driven, matched-marginal, H = 0.35 | 0.3518 | 0.3667 | 0.3687 | **+0.0169 [+0.0136, +0.0202]** | 48/50 | +0.0149 |

## Learner contrast (paired on the same windows)

| Row | Logistic Combined - Forest Combined [95% CI] | Logistic Statistics - Forest Statistics |
|---|---:|---:|
| Heston, scale-only | +0.0003 [-0.0024, +0.0030] | **-0.0050 [-0.0075, -0.0026]** |
| fBM-driven, scale-only, H = 0.2 | **-0.0061 [-0.0082, -0.0040]** | **-0.0068 [-0.0088, -0.0049]** |
| fBM-driven, scale-only, H = 0.35 | **-0.0040 [-0.0063, -0.0018]** | **-0.0062 [-0.0084, -0.0041]** |
| Heston, matched-marginal | **+0.0139 [+0.0095, +0.0183]** | +0.0026 [-0.0024, +0.0076] |
| fBM-driven, matched-marginal, H = 0.2 | +0.0019 [-0.0022, +0.0061] | -0.0009 [-0.0053, +0.0035] |
| fBM-driven, matched-marginal, H = 0.35 | **+0.0074 [+0.0028, +0.0120]** | **-0.0065 [-0.0111, -0.0019]** |

## Causal detector, Combined minus Statistics

Lower delay and fewer false switches are better. Delay uses replications with at least one matched change.

| Row | Learner | Detector BA | Boundary F1 | Detection prob. | Delay (obs.) | False switches / 1,000 |
|---|---|---:|---:|---:|---:|---:|
| Heston, scale-only | forest | **+0.0021 [+0.0005, +0.0038]** | -0.000 [-0.005, +0.005] | -0.007 [-0.018, +0.005] | -0.65 [-1.50, +0.19] | -0.09 [-0.23, +0.05] |
| Heston, scale-only | logistic | **+0.0078 [+0.0044, +0.0113]** | **+0.017 [+0.009, +0.026]** | **+0.037 [+0.012, +0.063]** | **-1.45 [-2.73, -0.18]** | +0.04 [-0.26, +0.34] |
| fBM-driven, scale-only, H = 0.2 | forest | +0.0005 [-0.0003, +0.0012] | +0.000 [-0.005, +0.005] | +0.001 [-0.009, +0.011] | -0.03 [-0.60, +0.54] | +0.00 [-0.08, +0.08] |
| fBM-driven, scale-only, H = 0.2 | logistic | **+0.0023 [+0.0005, +0.0042]** | **+0.010 [+0.000, +0.020]** | **+0.038 [+0.019, +0.058]** | **-2.35 [-3.41, -1.29]** | +0.16 [-0.04, +0.36] |
| fBM-driven, scale-only, H = 0.35 | forest | **+0.0010 [+0.0000, +0.0019]** | +0.003 [-0.003, +0.008] | -0.001 [-0.011, +0.009] | +0.09 [-0.60, +0.77] | -0.08 [-0.20, +0.03] |
| fBM-driven, scale-only, H = 0.35 | logistic | **+0.0057 [+0.0033, +0.0082]** | **+0.023 [+0.014, +0.032]** | **+0.055 [+0.037, +0.072]** | **-1.50 [-2.76, -0.23]** | +0.12 [-0.07, +0.31] |
| Heston, matched-marginal | forest | +0.0073 [-0.0025, +0.0171] | +0.001 [-0.008, +0.011] | -0.008 [-0.043, +0.027] | +0.45 [-2.27, +3.18] | -0.15 [-1.00, +0.71] |
| Heston, matched-marginal | logistic | **+0.0314 [+0.0187, +0.0442]** | +0.000 [-0.012, +0.013] | **-0.080 [-0.141, -0.019]** | +1.17 [-1.68, +4.02] | **-2.41 [-3.92, -0.90]** |
| fBM-driven, matched-marginal, H = 0.2 | forest | +0.0018 [-0.0063, +0.0099] | +0.006 [-0.003, +0.014] | -0.024 [-0.055, +0.007] | +0.22 [-2.81, +3.26] | **-1.10 [-2.07, -0.13]** |
| fBM-driven, matched-marginal, H = 0.2 | logistic | **+0.0114 [+0.0027, +0.0201]** | +0.002 [-0.006, +0.010] | **+0.073 [+0.032, +0.115]** | -1.77 [-4.76, +1.21] | **+2.02 [+0.84, +3.20]** |
| fBM-driven, matched-marginal, H = 0.35 | forest | -0.0021 [-0.0123, +0.0081] | -0.001 [-0.010, +0.008] | +0.012 [-0.021, +0.045] | **+2.68 [+0.20, +5.15]** | +0.46 [-0.43, +1.36] |
| fBM-driven, matched-marginal, H = 0.35 | logistic | **+0.0317 [+0.0216, +0.0418]** | +0.006 [-0.005, +0.017] | +0.034 [-0.018, +0.085] | **+2.77 [+0.06, +5.49]** | +0.11 [-1.25, +1.46] |

## Balanced accuracy by observations since the last true change, Combined minus Statistics

| Row | Learner | 0-24 | 25-49 | 50-99 | 100+ |
|---|---|---:|---:|---:|---:|
| Heston, scale-only | forest | -0.0008 [-0.0026, +0.0011] | +0.0013 [-0.0015, +0.0040] | +0.0019 [-0.0007, +0.0045] | **+0.0034 [+0.0027, +0.0042]** |
| Heston, scale-only | logistic | -0.0020 [-0.0065, +0.0025] | **+0.0286 [+0.0232, +0.0340]** | **+0.0286 [+0.0228, +0.0343]** | **+0.0055 [+0.0038, +0.0073]** |
| fBM-driven, scale-only, H = 0.2 | forest | +0.0010 [-0.0003, +0.0023] | **+0.0095 [+0.0072, +0.0119]** | **+0.0038 [+0.0017, +0.0060]** | +0.0004 [-0.0000, +0.0007] |
| fBM-driven, scale-only, H = 0.2 | logistic | **+0.0026 [+0.0002, +0.0049]** | **+0.0317 [+0.0264, +0.0371]** | **+0.0318 [+0.0271, +0.0365]** | **-0.0033 [-0.0045, -0.0022]** |
| fBM-driven, scale-only, H = 0.35 | forest | +0.0017 [-0.0000, +0.0034] | **+0.0062 [+0.0029, +0.0094]** | **+0.0027 [+0.0004, +0.0050]** | **+0.0009 [+0.0004, +0.0015]** |
| fBM-driven, scale-only, H = 0.35 | logistic | +0.0003 [-0.0023, +0.0029] | **+0.0292 [+0.0248, +0.0335]** | **+0.0327 [+0.0281, +0.0374]** | -0.0009 [-0.0021, +0.0002] |
| Heston, matched-marginal | forest | -0.0031 [-0.0086, +0.0024] | **+0.0075 [+0.0009, +0.0140]** | **+0.0131 [+0.0083, +0.0178]** | **+0.0074 [+0.0050, +0.0097]** |
| Heston, matched-marginal | logistic | **-0.0132 [-0.0247, -0.0017]** | **+0.0128 [+0.0041, +0.0214]** | **+0.0268 [+0.0184, +0.0352]** | **+0.0200 [+0.0165, +0.0235]** |
| fBM-driven, matched-marginal, H = 0.2 | forest | +0.0024 [-0.0036, +0.0083] | **+0.0068 [+0.0010, +0.0126]** | +0.0053 [-0.0003, +0.0108] | **+0.0021 [+0.0006, +0.0035]** |
| fBM-driven, matched-marginal, H = 0.2 | logistic | -0.0039 [-0.0131, +0.0052] | +0.0047 [-0.0064, +0.0159] | **+0.0096 [+0.0028, +0.0164]** | **+0.0056 [+0.0025, +0.0087]** |
| fBM-driven, matched-marginal, H = 0.35 | forest | -0.0039 [-0.0090, +0.0012] | +0.0016 [-0.0045, +0.0077] | **+0.0062 [+0.0015, +0.0109]** | **+0.0030 [+0.0009, +0.0051]** |
| fBM-driven, matched-marginal, H = 0.35 | logistic | **-0.0114 [-0.0224, -0.0004]** | **+0.0117 [+0.0006, +0.0229]** | **+0.0178 [+0.0100, +0.0257]** | **+0.0189 [+0.0150, +0.0228]** |

## Truncation-order ablation (Heston rows, `truncation_*/`)

Combined minus Statistics by log-signature truncation level; the level-3 column is the main-grid cell. The
noise control is Statistics plus 32 i.i.d. standard-normal columns (level-4 dimension).

| Design | Learner | Level 2 (6) | Level 3 (14) | Level 4 (32) | Step 2 -> 3 | Step 3 -> 4 | Noise control, level 4 |
|---|---|---:|---:|---:|---:|---:|---:|
| Heston, scale-only | forest | **+0.0039 [+0.0032, +0.0046]** | **+0.0029 [+0.0022, +0.0036]** | **+0.0021 [+0.0011, +0.0030]** | **-0.0010 [-0.0014, -0.0005]** | **-0.0008 [-0.0014, -0.0003]** | **-0.0017 [-0.0026, -0.0008]** |
| Heston, scale-only | logistic | **+0.0082 [+0.0071, +0.0093]** | **+0.0083 [+0.0068, +0.0097]** | **+0.0068 [+0.0054, +0.0082]** | +0.0001 [-0.0008, +0.0009] | **-0.0015 [-0.0020, -0.0009]** | **-0.0007 [-0.0011, -0.0003]** |
| Heston, matched-marginal | forest | **+0.0051 [+0.0033, +0.0068]** | **+0.0075 [+0.0053, +0.0097]** | **+0.0111 [+0.0079, +0.0143]** | **+0.0024 [+0.0013, +0.0036]** | **+0.0036 [+0.0019, +0.0053]** | +0.0000 [-0.0020, +0.0020] |
| Heston, matched-marginal | logistic | **+0.0074 [+0.0056, +0.0091]** | **+0.0188 [+0.0159, +0.0217]** | **+0.0222 [+0.0189, +0.0255]** | **+0.0115 [+0.0094, +0.0135]** | **+0.0034 [+0.0018, +0.0050]** | -0.0004 [-0.0010, +0.0003] |

## Reading across rows

- **Pointwise, forest.** Adding log-signatures to the 11 moment features raises test balanced accuracy in every
  row, by 0.001-0.008; the interval excludes zero everywhere. The gain is largest in the Heston matched-marginal
  row (variance time scale is the only regime marker) and smallest in the rough scale-only rows (where the
  variance level is an almost perfect marker and Statistics alone reaches 0.79-0.80). Signature alone never beats
  Statistics for the forest except, marginally, in the Heston matched-marginal row.
- **Pointwise, logistic.** The linear learner extracts a larger within-learner increment in every row. In the
  Heston matched-marginal row and the rough matched-marginal row at H = 0.35 it also reaches a higher final
  Combined accuracy than the forest (+0.014 and +0.007); in the Heston scale-only row the two learners tie
  (+0.0003); in the rough scale-only rows it ends below the forest.
  Which learner is "better" therefore depends on the design, and the signature's value is learner-dependent.
- **Where the gain lives.** For the forest the gain sits in windows lying entirely inside the new regime in the
  Heston rows and in the transition buckets in the rough scale-only rows; it is never resolved in the first 25
  observations after a switch. The logistic gain is concentrated at 25-99 observations after a switch in the
  scale-only rows and is negative at 0-24 in the Heston and H = 0.35 matched-marginal rows.
- **Event level.** For the forest, no detector contrast excludes zero consistently across rows (the archived
  scale-only delay reduction vanished with the path fix; the rough rows show isolated false-switch or delay
  effects of opposite sign). For the logistic learner the scale-only rows show higher boundary F1, higher
  detection probability and shorter delay at unchanged false-switch rates, while the matched-marginal rows show
  operating-point trade-offs (fewer false switches but fewer detections in the Heston row, longer delay at
  H = 0.35). A higher state score is not a consistently better transition signal.
- **Truncation level.** The two Heston designs pull in opposite directions. In the scale-only design the whole
  forest gain is in level 2 (increments and Lévy areas) and levels 3-4 subtract from it; in the matched-marginal
  design the gain grows with every level for both learners (forest +0.005 -> +0.011, logistic +0.007 ->
  +0.022 from level 2 to 4). The noise control is flat everywhere, so none of this is a dimension effect. What the
  signature contributes is therefore design-specific: level-2 timing of absolute returns when the variance level
  carries the regime, higher iterated integrals when only its time scale does.
- **Roughness.** With identical parameter families, antipersistent fGn shrinks the stationary variance dispersion
  to 18-52% of the Heston value and removes most return kurtosis. This makes the scale contrast easier (Statistics
  0.80 vs 0.74) and the time-scale contrast much harder (0.34-0.36 vs 0.40, near the 0.333 chance level), and
  the matched-marginal rows are no longer exactly matched across regimes. The direction of the signature effect
  survives in every row; its size does not transfer. The rows answer "does the fBM-increment approximation change
  the answer", not "does roughness per se".
