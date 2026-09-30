# Scale or speed? When do path signatures improve volatility-regime forecasts?

Code and replication materials for:

> Yang Lyu and Kiseop Lee, **“Scale or speed? When do path signatures improve
> volatility-regime forecasts?”**

This repository studies when path signatures, used as model-free features of return
paths, improve out-of-sample forecasts of volatility regimes. The forecasting study
predicts the tercile of next-month realized volatility in United States and
developed-ex-US equity markets, 2005-2025, and compares machine-learning classifiers
with and without signatures against conventional benchmarks, including a
heterogeneous autoregressive (HAR) regression that is also augmented with
signatures. Two controlled Heston experiments explain the results by separating
regimes that differ in variance **scale** from regimes that share the same stationary
variance law but differ in mean-reversion **speed**.

## Main findings

- The HAR regression is hard to beat: no classifier, with or without signatures,
  forecasts next-month volatility terciles more accurately.
- Adding raw signatures to the HAR regression improves United States forecasts
  (balanced accuracy +2.7 points and a lower ranked probability score, both resolved
  after Holm correction); adding conventional features does not, and no gain survives
  correction outside the United States.
- In the controlled Heston designs, signature gains are larger when regimes differ in
  speed than in variance level, most of all for raw signatures with a linear learner.
- Better regime classification does not yield earlier change detection at matched
  false-alarm rates.
- No accuracy gain translates into a resolved Sharpe-ratio improvement in a
  volatility-managed portfolio.

The repository contains the frozen replication-level outputs behind the reported
tables, along with scripts for rebuilding derived tables and figures.

## Study design

**Forecasting study.** Daily value-weighted market returns from the Kenneth French Data
Library, July 1990 to December 2025. On each day the target is the tercile of
annualized realized volatility over the next 21 trading days. Models are refitted
annually on an expanding window with a 21-day embargo and selected on the last three
training years; the out-of-sample period is 2005-2025. Benchmarks are persistence, a
tercile transition matrix and a log-HAR regression with Gaussian residuals; the HAR
regression is also augmented with ridge-penalized signature coordinates or, as a
control, the Statistics features. Forecasts are scored by balanced accuracy, the ranked
probability score, the logarithmic score and the Brier score, with Diebold-Mariano
tests, stationary-bootstrap intervals and Holm correction.

**Simulation study.** Each replication uses independent parameter draws and complete-path
train/validation/test splits. Causal 50-return windows are represented by:

- **Statistics:** 11 conventional return and volatility features;
- **B3:** Statistics plus 29 additional conventional features;
- **raw signature:** 39 level-3 coordinates;
- **log-signature:** 14 level-3 coordinates; and
- **HMM probabilities:** forward-filtered probabilities from a three-state
  Gaussian hidden Markov model.

The signature is computed on the origin-anchored, three-channel path

```text
(time, cumulative return, cumulative absolute return).
```

Random forests, multinomial logistic regression, and histogram gradient boosting
use prespecified three-setting grids selected on validation balanced accuracy.
Detector settings are also selected on validation paths and evaluated once on
held-out test paths.

## Repository structure

```text
.
├── regime_pipeline.py                 # simulation and causal feature extraction
├── regime_detection.py                # detector calibration and event matching
├── experiment_unified_grid.py         # main simulation comparison
├── experiment_matched_false_alarms.py # matched-false-alarm detector study
├── experiment_market_french.py        # walk-forward market study
├── experiment_truncation_ablation.py  # signature-level ablation
├── experiment_a_monte_carlo.py        # scale-design main-grid rows (Table S10); shared helpers
├── experiment_b_monte_carlo.py        # speed-design main-grid rows (Table S10)
├── market_forecast_evaluation.py      # scores and conventional benchmarks for the market forecasts
├── market_har_signatures.py           # HAR regression augmented with signatures
├── market_har_economics.py            # volatility-managed exposure from the HAR forecasts
├── market_economics_all_rules.py      # exposure results for the remaining signature rules
├── audit_discretization.py            # Euler discretization audit of Design B
├── multiplicity_holm.py               # Holm multiplicity correction
├── refit_unified_grid_hmm.py           # HMM-only refit utility
├── tests/                              # unit tests
├── results/                            # frozen outputs and result-specific READMEs
├── data/                               # local market-data archives (git-ignored)
└── paper/supplement/                   # supplementary tables and figure builders
```

The other top-level scripts (`beyond_marginal_benchmark.py`,
`experiment_b_window_stability.py`, `experiment_fbm_monte_carlo.py` and
`run_leakage_free_experiment.py`) come from earlier stages of the study and are kept
for provenance; the main scripts or the tests import the first three. Their outputs
are not reported in the paper.

The principal frozen outputs are:

```text
results/unified_grid/{scale_only,matched_marginal}/
results/unified_grid/detector_matched_fa/
results/unified_grid_logitc/
results/main_grid/truncation_{scale_only,matched_marginal}/
results/main_grid/heston_{scale_only,matched_marginal}/
results/market_french/
results/market_economics_all_rules/
results/market_forecast_evaluation/
results/market_har_signatures/
results/market_har_economics/
results/multiplicity/
results/discretization_audit/
```

Each result directory contains a configuration file and a local README describing
its protocol and outputs. CSV files contain the replication-level and summarized
results used in the manuscript.

## Environment

The frozen results were produced with Python 3.9.9. Create an isolated environment
from the repository root:

```bash
python3.9 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools
python -m pip install numpy==2.0.2
python -m pip install --no-build-isolation --no-deps \
  "iisignature @ git+https://github.com/bottler/iisignature@447577b55ad07a985c0a39b7c4fbf01ca2b142e5"
python -m pip install -r requirements.txt
```

`iisignature` is pinned to the commit used for the study. The PyPI 0.24 wheel did
not support the required log-signature preparation reliably on the development
platform.

Run the test suite with:

```bash
python -m unittest \
  tests.test_regime_pipeline \
  tests.test_unified_grid \
  tests.test_matched_false_alarms \
  tests.test_market_french \
  tests.test_multiplicity_holm
```

## Reproducing the analyses

The commands below write to `reproduced/` so that the frozen outputs under
`results/` are not overwritten.

### Main simulation grid

```bash
python experiment_unified_grid.py \
  --design scale_only \
  --jobs 8 \
  --output-dir reproduced/unified_grid/scale_only

python experiment_unified_grid.py \
  --design matched_marginal \
  --jobs 8 \
  --output-dir reproduced/unified_grid/matched_marginal
```

The defaults reproduce the paper protocol: 50 paired replications, 10 training
paths, 10 validation paths, five test paths, 5,000 observations per path, and a
50-observation causal window.

### Detection at matched false-alarm budgets

The detector study reads saved probabilities and simulated paths from the unified
grid; it does not refit the classifiers:

```bash
python experiment_matched_false_alarms.py \
  --grid-root results/unified_grid \
  --output-root reproduced/detector_matched_fa \
  --jobs 8
```

The large per-replication probability and path archives are not in Git history; they
are attached to release v1.0 as `unified_grid_archives.tar.gz` (SHA-256
`98478628e236d87600c4ae7c739a0d47b20cac3507fa50fd271b7400e83d4a7c`). Download and
extract them from the repository root:

```bash
curl -LO https://github.com/letdaly/regime_signature/releases/download/v1.0/unified_grid_archives.tar.gz
tar -xzf unified_grid_archives.tar.gz -C results/unified_grid
```

Without the archives, rerun the main grid first and point `--grid-root` to the new output.

### Truncation-order ablation

```bash
python experiment_truncation_ablation.py \
  --design scale_only \
  --jobs 8 \
  --output-dir reproduced/truncation_scale_only

python experiment_truncation_ablation.py \
  --design matched_marginal \
  --jobs 8 \
  --output-dir reproduced/truncation_matched_marginal
```

### Accuracy by time since the last change

Supplementary Table S10 uses the Heston rows of the earlier main grid (level-3
log-signatures, a fixed-depth random forest and a validation-tuned logistic
regression). The defaults reproduce the recorded configurations:

```bash
python experiment_a_monte_carlo.py \
  --jobs 8 \
  --output-dir reproduced/main_grid/heston_scale_only

python experiment_b_monte_carlo.py \
  --jobs 8 \
  --output-dir reproduced/main_grid/heston_matched_marginal
```

### Market study

The market study uses daily value-weighted United States and developed-ex-US
returns from the Kenneth French Data Library. Freeze the input files and their
SHA-256 hashes before fitting:

```bash
python experiment_market_french.py \
  --market both \
  --download \
  --freeze \
  --output-dir reproduced/market_french

python experiment_market_french.py \
  --market both \
  --output-dir reproduced/market_french \
  --model-jobs -1
```

The reported sample runs from July 1990 through December 2025, with annual
walk-forward tests from 2005 through 2025 and a 21-trading-day embargo.

### Multiplicity correction

The published multiplicity audit is rebuilt from the frozen outputs without
refitting:

```bash
python multiplicity_holm.py --output-dir reproduced/multiplicity
```

### Remaining signature rules and discretization audit

The volatility-managed results for the two signature rules that were not
prespecified are rebuilt from the stored out-of-sample weights, and the Euler
discretization of the Design B variance process is audited by simulation:

```bash
python market_economics_all_rules.py --output-dir reproduced/market_economics_all_rules
python audit_discretization.py --output-dir reproduced/discretization_audit
```

### Forecast evaluation and HAR benchmarks

The probabilistic evaluation of the market forecasts, with climatology, persistence,
transition and HAR benchmarks, rebuilds the market folds without refitting any classifier:

```bash
python market_forecast_evaluation.py --output-dir reproduced/market_forecast_evaluation
```

The HAR benchmark augmented with signature coordinates (and, as a control, with the
Statistics features) recomputes the market features but refits no classifier:

```bash
python market_har_signatures.py --output-dir reproduced/market_har_signatures
```

The same volatility-managed exposure rule, driven by the plain and augmented HAR forecasts:

```bash
python market_har_economics.py --output-dir reproduced/market_har_economics
```

These three scripts rebuild the market windows from the frozen Kenneth French
archives in `data/french` (see the market study above) and stop if the archives do
not match the SHA-256 hashes recorded in `results/market_french/<market>/config.json`.

### Supplementary tables and figure

```bash
python paper/supplement/build_supplement_tables.py
python paper/supplement/build_b3_figure.py
```

These scripts read the frozen outputs under `results/` and regenerate the LaTeX
table fragments and the year-by-year B3 figure.

## Data and output provenance

- Simulation paths are generated locally from recorded seeds and configuration
  files.
- Market returns are downloaded from the
  [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
- Each frozen market configuration records the input archive SHA-256 and sample
  definition.
- No test observation is used for feature, learner, hyperparameter, or detector
  selection.
- Large probability/path archives are data products rather than source code and are
  distributed through release v1.0 instead of Git history.
- The Kenneth French archives are not redistributed. The library revises its files,
  so a fresh download may not match the recorded hashes; the market scripts then stop
  rather than silently use different data.

## Citation

If you use this repository, please cite the paper and the specific GitHub release
used for your analysis. A `CITATION.cff` file and final bibliographic details will
be added with the archival release.

```text
Lyu, Yang, and Kiseop Lee. “Scale or speed? When do path signatures improve
volatility-regime forecasts?”
```

## License

Released under the MIT License; see `LICENSE`.

## Authors

- Yang Lyu, Department of Statistics, Purdue University
- Kiseop Lee, Department of Statistics, Purdue University

## Disclaimer

The market exercise is an empirical evaluation of predictive representations,
not a proposed trading strategy or investment advice.
