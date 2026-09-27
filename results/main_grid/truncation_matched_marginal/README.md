# Signature truncation-order ablation: matched-marginal design

## Configuration

Run 2026-09-20: 50 paired replications, 10 train / 10 validation / 5 test paths of 5,000 observations, 50-return
causal windows, same base seed as `results/main_grid/heston_matched_marginal` (identical parameter families and paths). One level-4 log-signature
matrix per replication (32 coordinates on the origin-anchored three-channel path); the level-2 and level-3
log-signatures are its first 6 and 14 columns, which is exact because iisignature orders log-signature
coordinates by level. `combined_3` is therefore the main-grid Combined cell: the maximum absolute difference in
test balanced accuracy over the 50 replications is 0.0e+00. The dimension control `noise_L` is Statistics
plus d_L i.i.d. standard-normal columns drawn independently for every window from the replication seed. Both
learners as in the main grid (Random Forest 200 trees / depth 6 / balanced; logistic regression with robust
scaling, tail clipping and C in {0.01, 0.1, 1} selected on validation). Pointwise classification and
time-since-change stratification only; the causal detector is not run.

Within each path all regimes share theta, mu, rho, and sigma^2/kappa; only kappa changes, so the stationary CIR variance law is identical across regimes. Four Euler substeps per observation.

## Random Forest

Statistics alone: 0.3952 (main grid 0.3952).

| Level | Coordinates | Signature | Combined | Statistics + noise | Combined - Statistics | Combined - noise | Signature - Statistics | Noise - Statistics |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 6 | 0.3935 | 0.4002 | 0.3951 | **+0.0051 [+0.0033, +0.0068]** (80%) | **+0.0051 [+0.0037, +0.0066]** | -0.0016 [-0.0072, +0.0040] | -0.0001 [-0.0013, +0.0012] |
| 3 | 14 | 0.3998 | 0.4027 | 0.3956 | **+0.0075 [+0.0053, +0.0097]** (86%) | **+0.0071 [+0.0053, +0.0090]** | +0.0047 [-0.0005, +0.0098] | +0.0004 [-0.0011, +0.0018] |
| 4 | 32 | 0.4034 | 0.4063 | 0.3952 | **+0.0111 [+0.0079, +0.0143]** (84%) | **+0.0111 [+0.0086, +0.0136]** | **+0.0082 [+0.0029, +0.0136]** | +0.0000 [-0.0020, +0.0020] |

| Step | Combined balanced-accuracy change [95% CI] | Positive |
|---|---:|---:|
| level 2 -> 3 | **+0.0024 [+0.0013, +0.0036]** | 68% |
| level 3 -> 4 | **+0.0036 [+0.0019, +0.0053]** | 74% |

Combined minus noise control by observations since the last true change:

| Bucket | L2: Combined - noise | L3: Combined - noise | L4: Combined - noise |
|---|---:|---:|---:|
| 0-24 | -0.0007 [-0.0054, +0.0041] | +0.0003 [-0.0055, +0.0061] | -0.0029 [-0.0105, +0.0046] |
| 25-49 | **+0.0063 [+0.0018, +0.0108]** | **+0.0081 [+0.0019, +0.0143]** | +0.0087 [-0.0004, +0.0179] |
| 50-99 | **+0.0063 [+0.0015, +0.0110]** | **+0.0102 [+0.0047, +0.0157]** | **+0.0129 [+0.0069, +0.0189]** |
| 100+ | **+0.0052 [+0.0036, +0.0068]** | **+0.0070 [+0.0050, +0.0090]** | **+0.0118 [+0.0091, +0.0145]** |

## Logistic regression

Statistics alone: 0.3978. Selected regularization: statistics: C=0.01 x15, C=0.1 x9, C=1 x26; combined_2: C=0.01 x11, C=0.1 x6, C=1 x33; combined_3: C=0.01 x12, C=0.1 x17, C=1 x21; combined_4: C=0.01 x11, C=0.1 x14, C=1 x25.

| Level | Coordinates | Signature | Combined | Statistics + noise | Combined - Statistics | Combined - noise | Signature - Statistics | Noise - Statistics |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 6 | 0.3606 | 0.4051 | 0.3977 | **+0.0074 [+0.0056, +0.0091]** (88%) | **+0.0074 [+0.0056, +0.0092]** | **-0.0372 [-0.0452, -0.0292]** | -0.0001 [-0.0005, +0.0004] |
| 3 | 14 | 0.3953 | 0.4166 | 0.3978 | **+0.0188 [+0.0159, +0.0217]** (98%) | **+0.0188 [+0.0160, +0.0217]** | -0.0025 [-0.0091, +0.0041] | -0.0000 [-0.0006, +0.0005] |
| 4 | 32 | 0.4010 | 0.4200 | 0.3974 | **+0.0222 [+0.0189, +0.0255]** (100%) | **+0.0226 [+0.0194, +0.0257]** | +0.0032 [-0.0032, +0.0096] | -0.0004 [-0.0010, +0.0003] |

| Step | Combined balanced-accuracy change [95% CI] | Positive |
|---|---:|---:|
| level 2 -> 3 | **+0.0115 [+0.0094, +0.0135]** | 96% |
| level 3 -> 4 | **+0.0034 [+0.0018, +0.0050]** | 72% |

Combined minus noise control by observations since the last true change:

| Bucket | L2: Combined - noise | L3: Combined - noise | L4: Combined - noise |
|---|---:|---:|---:|
| 0-24 | **-0.0087 [-0.0158, -0.0017]** | **-0.0137 [-0.0249, -0.0025]** | **-0.0177 [-0.0297, -0.0057]** |
| 25-49 | +0.0010 [-0.0052, +0.0072] | **+0.0111 [+0.0019, +0.0203]** | **+0.0101 [+0.0003, +0.0198]** |
| 50-99 | **+0.0154 [+0.0099, +0.0209]** | **+0.0272 [+0.0196, +0.0347]** | **+0.0304 [+0.0224, +0.0384]** |
| 100+ | **+0.0079 [+0.0057, +0.0101]** | **+0.0201 [+0.0167, +0.0236]** | **+0.0245 [+0.0205, +0.0286]** |

## Reading

- **The gain grows with the truncation level for both learners.** Forest: **+0.0051 [+0.0033, +0.0068]** at level 2, **+0.0075 [+0.0053, +0.0097]** at level 3,
  **+0.0111 [+0.0079, +0.0143]** at level 4, with both steps excluding zero (**+0.0024 [+0.0013, +0.0036]**, **+0.0036 [+0.0019, +0.0053]**). Logistic: **+0.0074 [+0.0056, +0.0091]**, **+0.0188 [+0.0159, +0.0217]**, **+0.0222 [+0.0189, +0.0255]**; the level
  2 -> 3 step is the big one (**+0.0115 [+0.0094, +0.0135]**) and 3 -> 4 still adds (**+0.0034 [+0.0018, +0.0050]**). Where the regimes differ only in the
  variance time scale, the information sits in the higher iterated integrals, the opposite of the scale-only
  design.
- **Not a dimension effect.** Statistics plus matching noise columns is flat (-0.0001 [-0.0013, +0.0012] / +0.0004 [-0.0011, +0.0018] / +0.0000 [-0.0020, +0.0020]), so Combined
  minus noise equals Combined minus Statistics at every level (**+0.0051 [+0.0037, +0.0066]** / **+0.0071 [+0.0053, +0.0090]** / **+0.0111 [+0.0086, +0.0136]**).
- **Signature alone.** For the forest level 4 alone beats Statistics (**+0.0082 [+0.0029, +0.0136]**); for the logistic learner
  Signature alone never does (level 2 is far below Statistics, **-0.0372 [-0.0452, -0.0292]**; level 4 +0.0032 [-0.0032, +0.0096]), so the linear model's
  large Combined gain needs the moment features as well.
- **Where it lives.** Forest gains over the noise control grow with the level in windows inside the new regime
  (100+: **+0.0052 [+0.0036, +0.0068]** -> **+0.0118 [+0.0091, +0.0145]**); the logistic learner's gain is largest at 50-99 (**+0.0304 [+0.0224, +0.0384]**) and 100+
  (**+0.0245 [+0.0205, +0.0286]**) and increasingly *negative* at 0-24 (**-0.0087 [-0.0158, -0.0017]** -> **-0.0177 [-0.0297, -0.0057]**): more signature coordinates make the
  linear model slower to leave the previous state.

## Output files

`config.json`, `replications.csv`, `summary.csv` (with `positive_share` for every paired contrast),
`test_diagnostics.csv` (tidy: learner x representation, recalls, stratified BA, selected C).
