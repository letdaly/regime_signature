# Signature truncation-order ablation: scale-only design

## Configuration

Run 2026-09-20: 50 paired replications, 10 train / 10 validation / 5 test paths of 5,000 observations, 50-return
causal windows, same base seed as `results/main_grid/heston_scale_only` (identical parameter families and paths). One level-4 log-signature
matrix per replication (32 coordinates on the origin-anchored three-channel path); the level-2 and level-3
log-signatures are its first 6 and 14 columns, which is exact because iisignature orders log-signature
coordinates by level. `combined_3` is therefore the main-grid Combined cell: the maximum absolute difference in
test balanced accuracy over the 50 replications is 0.0e+00. The dimension control `noise_L` is Statistics
plus d_L i.i.d. standard-normal columns drawn independently for every window from the replication seed. Both
learners as in the main grid (Random Forest 200 trees / depth 6 / balanced; logistic regression with robust
scaling, tail clipping and C in {0.01, 0.1, 1} selected on validation). Pointwise classification and
time-since-change stratification only; the causal detector is not run.

Within each path all regimes share mu, kappa, rho, and c; only theta_i changes, sigma_i = c sqrt(kappa theta_i). One Euler step per observation.

## Random Forest

Statistics alone: 0.7353 (main grid 0.7353).

| Level | Coordinates | Signature | Combined | Statistics + noise | Combined - Statistics | Combined - noise | Signature - Statistics | Noise - Statistics |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 6 | 0.7352 | 0.7392 | 0.7353 | **+0.0039 [+0.0032, +0.0046]** (94%) | **+0.0040 [+0.0032, +0.0047]** | -0.0001 [-0.0022, +0.0019] | -0.0001 [-0.0004, +0.0003] |
| 3 | 14 | 0.7306 | 0.7382 | 0.7346 | **+0.0029 [+0.0022, +0.0036]** (88%) | **+0.0037 [+0.0030, +0.0043]** | **-0.0047 [-0.0068, -0.0026]** | **-0.0007 [-0.0012, -0.0002]** |
| 4 | 32 | 0.7283 | 0.7374 | 0.7336 | **+0.0021 [+0.0011, +0.0030]** (68%) | **+0.0038 [+0.0029, +0.0046]** | **-0.0070 [-0.0091, -0.0049]** | **-0.0017 [-0.0026, -0.0008]** |

| Step | Combined balanced-accuracy change [95% CI] | Positive |
|---|---:|---:|
| level 2 -> 3 | **-0.0010 [-0.0014, -0.0005]** | 26% |
| level 3 -> 4 | **-0.0008 [-0.0014, -0.0003]** | 32% |

Combined minus noise control by observations since the last true change:

| Bucket | L2: Combined - noise | L3: Combined - noise | L4: Combined - noise |
|---|---:|---:|---:|
| 0-24 | -0.0011 [-0.0034, +0.0012] | -0.0009 [-0.0029, +0.0010] | -0.0006 [-0.0034, +0.0021] |
| 25-49 | +0.0009 [-0.0021, +0.0038] | +0.0012 [-0.0013, +0.0038] | **+0.0070 [+0.0037, +0.0103]** |
| 50-99 | **+0.0062 [+0.0032, +0.0093]** | **+0.0044 [+0.0010, +0.0077]** | **+0.0055 [+0.0021, +0.0089]** |
| 100+ | **+0.0043 [+0.0034, +0.0052]** | **+0.0040 [+0.0033, +0.0048]** | **+0.0037 [+0.0028, +0.0046]** |

## Logistic regression

Statistics alone: 0.7303. Selected regularization: statistics: C=0.01 x10, C=0.1 x26, C=1 x14; combined_2: C=0.01 x14, C=0.1 x20, C=1 x16; combined_3: C=0.01 x12, C=0.1 x18, C=1 x20; combined_4: C=0.01 x13, C=0.1 x20, C=1 x17.

| Level | Coordinates | Signature | Combined | Statistics + noise | Combined - Statistics | Combined - noise | Signature - Statistics | Noise - Statistics |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 6 | 0.7353 | 0.7385 | 0.7298 | **+0.0082 [+0.0071, +0.0093]** (94%) | **+0.0086 [+0.0074, +0.0098]** | **+0.0051 [+0.0028, +0.0073]** | **-0.0004 [-0.0008, -0.0001]** |
| 3 | 14 | 0.7355 | 0.7385 | 0.7298 | **+0.0083 [+0.0068, +0.0097]** (92%) | **+0.0088 [+0.0073, +0.0103]** | **+0.0052 [+0.0030, +0.0074]** | **-0.0005 [-0.0009, -0.0001]** |
| 4 | 32 | 0.7340 | 0.7371 | 0.7296 | **+0.0068 [+0.0054, +0.0082]** (88%) | **+0.0075 [+0.0060, +0.0090]** | **+0.0037 [+0.0013, +0.0060]** | **-0.0007 [-0.0011, -0.0003]** |

| Step | Combined balanced-accuracy change [95% CI] | Positive |
|---|---:|---:|
| level 2 -> 3 | +0.0001 [-0.0008, +0.0009] | 46% |
| level 3 -> 4 | **-0.0015 [-0.0020, -0.0009]** | 26% |

Combined minus noise control by observations since the last true change:

| Bucket | L2: Combined - noise | L3: Combined - noise | L4: Combined - noise |
|---|---:|---:|---:|
| 0-24 | -0.0015 [-0.0052, +0.0022] | -0.0016 [-0.0060, +0.0029] | -0.0014 [-0.0063, +0.0035] |
| 25-49 | **+0.0204 [+0.0156, +0.0252]** | **+0.0277 [+0.0223, +0.0331]** | **+0.0340 [+0.0280, +0.0401]** |
| 50-99 | **+0.0266 [+0.0214, +0.0318]** | **+0.0280 [+0.0226, +0.0335]** | **+0.0310 [+0.0248, +0.0371]** |
| 100+ | **+0.0067 [+0.0053, +0.0081]** | **+0.0062 [+0.0045, +0.0080]** | **+0.0040 [+0.0022, +0.0059]** |

## Reading

- **Everything the forest gains is in level 2.** Six coordinates (the three channel increments and the three
  Lévy areas) give **+0.0039 [+0.0032, +0.0046]**; adding level 3 lowers the gain by **-0.0010 [-0.0014, -0.0005]** and level 4 by a further **-0.0008 [-0.0014, -0.0003]**. Level-1
  already contains sum |r| and the level-2 area between time and cumulative |r| records *when* in the window the
  absolute returns were large; higher iterated integrals add nothing a variance-level contrast needs and cost the
  depth-6 forest a little.
- **Not a dimension effect.** Statistics plus matching noise columns never helps (-0.0001 [-0.0004, +0.0003] / **-0.0007 [-0.0012, -0.0002]** / **-0.0017 [-0.0026, -0.0008]**), so
  Combined minus noise is about +0.004 at every level (**+0.0040 [+0.0032, +0.0047]** / **+0.0037 [+0.0030, +0.0043]** / **+0.0038 [+0.0029, +0.0046]**). Signature alone equals Statistics
  at level 2 (-0.0001 [-0.0022, +0.0019]) and falls below it at levels 3-4.
- **The logistic learner agrees**: **+0.0082 [+0.0071, +0.0093]** at level 2, **+0.0083 [+0.0068, +0.0097]** at level 3, **+0.0068 [+0.0054, +0.0082]** at level 4, with the level 3 -> 4
  step negative (**-0.0015 [-0.0020, -0.0009]**). For the linear model Signature alone beats Statistics at every level (**+0.0051 [+0.0028, +0.0073]** at level 2).
- **Where it lives.** For the forest the level-2 gain over the noise control is in the 50-99 and 100+ buckets
  (**+0.0062 [+0.0032, +0.0093]**, **+0.0043 [+0.0034, +0.0052]**); for the logistic learner it is largest at 25-99 observations after a switch
  (**+0.0204 [+0.0156, +0.0252]**, **+0.0266 [+0.0214, +0.0318]**) and smaller in mature windows (**+0.0067 [+0.0053, +0.0081]**).

## Output files

`config.json`, `replications.csv`, `summary.csv` (with `positive_share` for every paired contrast),
`test_diagnostics.csv` (tidy: learner x representation, recalls, stratified BA, selected C).
