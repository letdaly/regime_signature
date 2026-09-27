"""Unified comparison grid behind the paper's simulation results.

One protocol for every cell of the paper's Table 1: 50 paired replications,
10 / 10 / 5 training / validation / test paths of 5,000 observations, 50-return
causal windows labelled by their endpoint state, level-3 signatures on the
origin-anchored three-channel path.  Within a replication the paths are
simulated once and one feature matrix holds every block:

* ``b1_`` / ``b2_`` / ``b3_``: the 40-feature conventional hierarchy of the
  mechanism benchmark; the 11-feature Statistics block of the main grid is a
  named subset of it (same formulas, checked by the tests);
* ``signature_``: the 39 raw level-3 coordinates; ``logsignature_``: the 14
  log-signature coordinates;
* ``shuffled_signature_`` / ``shuffled_logsignature_``: the same coordinates
  after permuting the increments inside each window (order placebo; raw and
  log placebos share the permutation);
* ``hmm_``: forward-filtered state probabilities of a three-state Gaussian
  hidden Markov model fitted on training returns and state-matched to labels
  on training windows (the Markov-switching benchmark).

Every cell is a column subset of that matrix.  Each learner (random forest,
multinomial logistic regression, histogram gradient boosting) selects one
setting from a three-point grid on validation balanced accuracy and is scored
once on the test windows.  Validation and test probabilities of the Table-1
cells are saved with the simulated paths for the matched-false-alarm detector
study (experiment_matched_false_alarms.py), so no cell has to be refitted there.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from threadpoolctl import threadpool_limits

from beyond_marginal_benchmark import extract_from_existing_paths
from experiment_a_monte_carlo import (
    LOGISTIC_C_GRID,
    classification_diagnostics,
    clip_scaled_features,
    guard_output_directory,
    mean_t_interval,
    replication_seeds,
    split_class_counts,
)
from regime_detection import REGIME_IDS, ordered_probabilities
from regime_pipeline import (
    CausalFeatureExtractor,
    PathLevelSplits,
    SimulatedPath,
    WindowDataset,
    build_path_level_splits,
    sample_matched_marginal_parameters,
    sample_scale_only_parameters,
)


# Base seeds equal the main-grid rows', so replication k simulates the same
# paths as results/main_grid/heston_<design> and the logistic Statistics cells
# reproduce those numbers exactly.
DESIGNS = {
    "scale_only": {
        "sampler": sample_scale_only_parameters,
        "substeps": 1,
        "base_seed": 20260827,
    },
    "matched_marginal": {
        "sampler": sample_matched_marginal_parameters,
        "substeps": 4,
        "base_seed": 20260828,
    },
}

# The main grid's 11-feature Statistics block, by its names inside the rich
# b1/b2/b3 hierarchy (same order as CausalFeatureExtractor's legacy block).
STATISTICS_COLUMNS = (
    "b1_realized_volatility_long",
    "b1_realized_volatility_short",
    "b1_mean_absolute_return",
    "b1_downside_semivariance",
    "b1_upside_semivariance",
    "b1_skewness",
    "b1_kurtosis",
    "b2_return_autocorrelation_1",
    "b2_absolute_return_autocorrelation_1",
    "b2_squared_return_autocorrelation_1",
    "b1_maximum_log_drawdown",
)
BLOCK_PREFIXES = {
    "b3": ("b1_", "b2_", "b3_"),
    "rawsig": ("signature_",),
    "logsig": ("logsignature_",),
    "rawsig_shuffled": ("shuffled_signature_",),
    "logsig_shuffled": ("shuffled_logsignature_",),
    "hmm": ("hmm_",),
}
BLOCKS = ("statistics",) + tuple(BLOCK_PREFIXES)

# Cells are ordered unions of blocks.  The first seven are the Table-1 rows;
# the rest are the signature-alone levels and the order placebos.
CELLS: Dict[str, Tuple[str, ...]] = {
    "hmm": ("hmm",),
    "statistics": ("statistics",),
    "statistics_logsig": ("statistics", "logsig"),
    "statistics_rawsig": ("statistics", "rawsig"),
    "b3": ("b3",),
    "b3_logsig": ("b3", "logsig"),
    "b3_rawsig": ("b3", "rawsig"),
    "logsig": ("logsig",),
    "rawsig": ("rawsig",),
    "logsig_shuffled": ("logsig_shuffled",),
    "rawsig_shuffled": ("rawsig_shuffled",),
    "statistics_logsig_shuffled": ("statistics", "logsig_shuffled"),
    "statistics_rawsig_shuffled": ("statistics", "rawsig_shuffled"),
    "b3_logsig_shuffled": ("b3", "logsig_shuffled"),
    "b3_rawsig_shuffled": ("b3", "rawsig_shuffled"),
}
TABLE_CELLS = (
    "hmm",
    "statistics",
    "statistics_logsig",
    "statistics_rawsig",
    "b3",
    "b3_logsig",
    "b3_rawsig",
)
# Paired within-learner contrasts (minuend, subtrahend).
CONTRASTS = (
    ("statistics_logsig", "statistics"),
    ("statistics_rawsig", "statistics"),
    ("b3_logsig", "b3"),
    ("b3_rawsig", "b3"),
    ("statistics_logsig_shuffled", "statistics"),
    ("statistics_rawsig_shuffled", "statistics"),
    ("b3_logsig_shuffled", "b3"),
    ("b3_rawsig_shuffled", "b3"),
    ("logsig", "logsig_shuffled"),
    ("rawsig", "rawsig_shuffled"),
    ("statistics_logsig", "statistics_logsig_shuffled"),
    ("statistics_rawsig", "statistics_rawsig_shuffled"),
    ("b3_logsig", "b3_logsig_shuffled"),
    ("b3_rawsig", "b3_rawsig_shuffled"),
    ("statistics_rawsig", "statistics_logsig"),
    ("b3_rawsig", "b3_logsig"),
    ("b3", "statistics"),
    ("hmm", "statistics"),
)
LEARNERS = ("random_forest", "logistic", "gbm")
LEARNER_CONTRASTS = (("logistic", "random_forest"), ("gbm", "random_forest"))
LEARNER_SHORT = {"random_forest": "RF", "logistic": "logit", "gbm": "GBM"}
# Equal three-point search budgets, selected on validation balanced accuracy.
RF_DEPTH_GRID = (4, 6, 10)
GBM_LEAF_GRID = (7, 15, 31)
GBM_LEARNING_RATE = 0.05
GBM_MAX_ITER = 300
HMM_STATES = len(REGIME_IDS)
HMM_RETURN_SCALE = 100.0  # percent returns keep the covariance floor negligible
HMM_MIN_COVAR = 1e-6
HMM_MAX_ITER = 200
# EM restarts: one persistent initialization (diagonal 0.97, zero means, state
# standard deviations at 0.5 / 1 / 2 times the pooled one) plus random
# k-means initializations; the fit with the highest training log-likelihood
# is kept.  A single default initialization lands in a lower-likelihood
# optimum with a short-lived outlier state in about half of the Design A
# replications.
HMM_RESTARTS = 5
HMM_PERSISTENT_DIAGONAL = 0.97
PROBABILITY_QUANTUM = np.iinfo(np.uint16).max


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------


def block_columns(feature_names: Iterable[str]) -> Dict[str, np.ndarray]:
    """Column indices of every block present in one combined feature matrix."""

    names = tuple(feature_names)
    position = {name: index for index, name in enumerate(names)}
    missing = [name for name in STATISTICS_COLUMNS if name not in position]
    if missing:
        raise ValueError("Feature matrix lacks Statistics columns: {}".format(missing))
    selections = {
        "statistics": np.asarray([position[name] for name in STATISTICS_COLUMNS], dtype=int)
    }
    for block, prefixes in BLOCK_PREFIXES.items():
        selections[block] = np.asarray(
            [index for index, name in enumerate(names) if name.startswith(prefixes)],
            dtype=int,
        )
        if len(selections[block]) == 0:
            del selections[block]
    return selections


def cell_columns(
    selections: Dict[str, np.ndarray], cells: Iterable[str] = tuple(CELLS)
) -> Dict[str, np.ndarray]:
    """Column indices of every requested cell (blocks concatenated in order)."""

    columns = {}
    for cell in cells:
        missing = [block for block in CELLS[cell] if block not in selections]
        if missing:
            raise ValueError("Feature matrix lacks the {} block(s) of {}.".format(missing, cell))
        columns[cell] = np.concatenate([selections[block] for block in CELLS[cell]])
    return columns


def _signature_extractor(window_size: int, kind: str, shuffled: bool, seed: int):
    return CausalFeatureExtractor(
        window_size=window_size,
        short_window=min(10, window_size),
        signature_level=3,
        representation="signature",
        signature_kind=kind,
        increment_order="shuffle" if shuffled else "original",
        order_seed=seed,
    )


def _append_columns(
    base: WindowDataset, extra: WindowDataset, prefix: str = ""
) -> WindowDataset:
    if not (
        np.array_equal(base.path_ids, extra.path_ids)
        and np.array_equal(base.end_times, extra.end_times)
    ):
        raise AssertionError("Window alignment differs between extractors.")
    return WindowDataset(
        X=np.concatenate((base.X, extra.X), axis=1),
        y=base.y,
        path_ids=base.path_ids,
        end_times=base.end_times,
        feature_names=base.feature_names
        + tuple(prefix + name for name in extra.feature_names),
    )


def _with_columns(
    splits: PathLevelSplits, extra: PathLevelSplits, prefix: str = ""
) -> PathLevelSplits:
    return PathLevelSplits(
        train=_append_columns(splits.train, extra.train, prefix),
        validation=_append_columns(splits.validation, extra.validation, prefix),
        test=_append_columns(splits.test, extra.test, prefix),
        paths=splits.paths,
    )


def build_grid_splits(
    design: str,
    seed: int,
    n_paths: Tuple[int, int, int],
    n_steps: int,
    window_size: int,
    p_switch: float,
) -> PathLevelSplits:
    """Simulate one replication and assemble every non-HMM block on its windows."""

    if design not in DESIGNS:
        raise ValueError("design must be one of {}.".format(tuple(DESIGNS)))
    base_extractor = CausalFeatureExtractor(
        window_size=window_size,
        short_window=min(10, window_size),
        signature_level=3,
        representation="combined",
        signature_kind="raw",
        statistics_level="rich",
    )
    splits = build_path_level_splits(
        extractor=base_extractor,
        parameter_sampler=DESIGNS[design]["sampler"],
        n_paths=n_paths,
        n_steps=n_steps,
        master_seed=seed,
        p_switch=p_switch,
        min_dwell=window_size,
        stationary_initial_variance=True,
        simulation_substeps=DESIGNS[design]["substeps"],
    )
    for kind, shuffled, prefix in (
        ("log", False, ""),
        ("raw", True, "shuffled_"),
        ("log", True, "shuffled_"),
    ):
        extra = extract_from_existing_paths(
            splits, _signature_extractor(window_size, kind, shuffled, seed)
        )
        splits = _with_columns(splits, extra, prefix)
    splits.assert_disjoint()
    return splits


# ---------------------------------------------------------------------------
# Markov-switching benchmark block
# ---------------------------------------------------------------------------


def _path_returns(path: SimulatedPath) -> np.ndarray:
    return path.frame["LogReturn"].to_numpy(dtype=float) * HMM_RETURN_SCALE


def _hmm_candidate(seed: int, persistent: bool, pooled_std: float) -> GaussianHMM:
    model = GaussianHMM(
        n_components=HMM_STATES,
        covariance_type="diag",
        min_covar=HMM_MIN_COVAR,
        n_iter=HMM_MAX_ITER,
        tol=1e-4,
        random_state=seed,
        init_params="" if persistent else "stmc",
    )
    if persistent:
        off_diagonal = (1.0 - HMM_PERSISTENT_DIAGONAL) / (HMM_STATES - 1)
        model.startprob_ = np.full(HMM_STATES, 1.0 / HMM_STATES)
        model.transmat_ = np.full((HMM_STATES, HMM_STATES), off_diagonal) + np.eye(HMM_STATES) * (
            HMM_PERSISTENT_DIAGONAL - off_diagonal
        )
        model.means_ = np.zeros((HMM_STATES, 1))
        model.covars_ = (pooled_std * np.array([0.5, 1.0, 2.0]))[:, None] ** 2
    return model


def fit_hmm(
    train_paths: Sequence[SimulatedPath], seed: int, restarts: int = HMM_RESTARTS
) -> Tuple[GaussianHMM, Dict[str, float]]:
    """Fit a Gaussian HMM to the training paths' (percent) returns by EM.

    ``restarts`` EM runs (the first from the persistent initialization, the
    rest from hmmlearn's random k-means initialization) are compared by their
    training log-likelihood and the best one is returned with the
    log-likelihood, the winning initialization and the number of runs.
    """

    if restarts < 1:
        raise ValueError("restarts must be positive.")
    sequences = [_path_returns(path) for path in train_paths]
    observations = np.concatenate(sequences)[:, None]
    lengths = [len(sequence) for sequence in sequences]
    pooled_std = float(np.std(observations))
    best = None
    for restart in range(restarts):
        model = _hmm_candidate(seed + restart, persistent=(restart == 0), pooled_std=pooled_std)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(observations, lengths=lengths)
            log_likelihood = float(model.score(observations, lengths=lengths))
        if best is None or log_likelihood > best[1]:
            best = (model, log_likelihood, restart)
    model, log_likelihood, restart = best
    return model, {
        "hmm_log_likelihood": log_likelihood,
        "hmm_selected_restart": float(restart),
        "hmm_restarts": float(restarts),
        "hmm_converged": float(model.monitor_.converged),
    }


def forward_filter(
    start_probabilities: np.ndarray,
    transition_matrix: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    """Causal state probabilities P(state_t | y_1..y_t) of a Gaussian HMM."""

    observations = np.asarray(observations, dtype=float)
    log_emission = norm.logpdf(
        observations[:, None], loc=means[None, :], scale=np.sqrt(variances)[None, :]
    )
    with np.errstate(divide="ignore"):
        log_transition = np.log(transition_matrix)
        log_alpha = np.log(start_probabilities) + log_emission[0]
    filtered = np.empty((len(observations), len(means)), dtype=float)
    log_alpha -= logsumexp(log_alpha)
    filtered[0] = np.exp(log_alpha)
    for step in range(1, len(observations)):
        log_alpha = logsumexp(log_alpha[:, None] + log_transition, axis=0) + log_emission[step]
        log_alpha -= logsumexp(log_alpha)
        filtered[step] = np.exp(log_alpha)
    return filtered


def hmm_parameters(model: GaussianHMM) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    variances = np.asarray(model.covars_, dtype=float).reshape(HMM_STATES, -1)[:, 0]
    return (
        np.asarray(model.startprob_, dtype=float),
        np.asarray(model.transmat_, dtype=float),
        np.asarray(model.means_, dtype=float)[:, 0],
        variances,
    )


def match_states(hidden_argmax: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return ``order`` with ``order[label] = hidden state`` maximizing agreement."""

    agreement = confusion_matrix(labels, hidden_argmax, labels=REGIME_IDS)
    label_rows, hidden_columns = linear_sum_assignment(-agreement)
    order = np.empty(HMM_STATES, dtype=int)
    order[label_rows] = hidden_columns
    return order


def hmm_block(
    splits: PathLevelSplits, seed: int, window_size: int
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Filtered state probabilities at every window endpoint, matched to labels.

    Returns ``{split: (n_windows, 3) array}`` aligned with ``splits.<split>``
    and diagnostics of the fitted filter (its own balanced accuracies, state
    standard deviations, and persistence).
    """

    model, diagnostics = fit_hmm(splits.paths["train"], seed)
    start, transition, means, variances = hmm_parameters(model)
    filtered: Dict[str, np.ndarray] = {}
    for split_name in ("train", "validation", "test"):
        dataset = getattr(splits, split_name)
        pieces = []
        for path in splits.paths[split_name]:
            probabilities = forward_filter(
                start, transition, means, variances, _path_returns(path)
            )[window_size - 1 :]
            window_rows = np.flatnonzero(dataset.path_ids == path.path_id)
            if len(window_rows) != len(probabilities) or not np.array_equal(
                dataset.end_times[window_rows], np.arange(window_size - 1, len(path.frame))
            ):
                raise AssertionError("HMM filter rows do not align with windows.")
            pieces.append((window_rows, probabilities))
        block = np.empty((len(dataset), HMM_STATES), dtype=float)
        for window_rows, probabilities in pieces:
            block[window_rows] = probabilities
        filtered[split_name] = block
    order = match_states(np.argmax(filtered["train"], axis=1), splits.train.y)
    matched = {split_name: block[:, order] for split_name, block in filtered.items()}
    for split_name in ("train", "validation", "test"):
        diagnostics["hmm_direct_{}_ba".format(split_name)] = float(
            balanced_accuracy_score(
                getattr(splits, split_name).y, np.argmax(matched[split_name], axis=1)
            )
        )
    for label in REGIME_IDS:
        diagnostics["hmm_state_std_{}".format(label)] = float(np.sqrt(variances[order[label]]))
        diagnostics["hmm_state_mean_{}".format(label)] = float(means[order[label]])
        diagnostics["hmm_persistence_{}".format(label)] = float(
            transition[order[label], order[label]]
        )
    return matched, diagnostics


def add_hmm_block(
    splits: PathLevelSplits, seed: int, window_size: int
) -> Tuple[PathLevelSplits, Dict[str, float]]:
    matched, diagnostics = hmm_block(splits, seed, window_size)
    names = tuple("hmm_{}".format(label) for label in REGIME_IDS)

    def extend(dataset: WindowDataset, block: np.ndarray) -> WindowDataset:
        return WindowDataset(
            X=np.concatenate((dataset.X, block), axis=1),
            y=dataset.y,
            path_ids=dataset.path_ids,
            end_times=dataset.end_times,
            feature_names=dataset.feature_names + names,
        )

    return (
        PathLevelSplits(
            train=extend(splits.train, matched["train"]),
            validation=extend(splits.validation, matched["validation"]),
            test=extend(splits.test, matched["test"]),
            paths=splits.paths,
        ),
        diagnostics,
    )


# ---------------------------------------------------------------------------
# Learners
# ---------------------------------------------------------------------------


def learner_candidates(
    learner: str,
    seed: int,
    n_estimators: int,
    model_jobs: int,
    logistic_c_grid: Sequence[float] = LOGISTIC_C_GRID,
) -> Iterator[Tuple[str, float, object]]:
    """Yield ``(setting label, setting value, unfitted estimator)`` for one learner."""

    if learner == "random_forest":
        for depth in RF_DEPTH_GRID:
            yield "max_depth={}".format(depth), float(depth), RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=depth,
                class_weight="balanced",
                random_state=seed,
                n_jobs=model_jobs,
            )
    elif learner == "logistic":
        # With the default grid these are experiment_a_monte_carlo.fit_learner's
        # candidates, so the logistic Statistics cells reproduce the main grid.
        # --logistic-c-grid widens the penalty search for the B3-versus-
        # Statistics audit; config.json records whichever grid was used.
        for c_value in logistic_c_grid:
            yield "C={}".format(c_value), float(c_value), Pipeline(
                [
                    ("scale", RobustScaler(quantile_range=(10.0, 90.0), unit_variance=True)),
                    ("clip", FunctionTransformer(clip_scaled_features)),
                    (
                        "model",
                        LogisticRegression(
                            C=c_value,
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=seed,
                        ),
                    ),
                ]
            )
    elif learner == "gbm":
        for leaves in GBM_LEAF_GRID:
            yield "max_leaf_nodes={}".format(leaves), float(leaves), HistGradientBoostingClassifier(
                max_leaf_nodes=leaves,
                learning_rate=GBM_LEARNING_RATE,
                max_iter=GBM_MAX_ITER,
                early_stopping=False,
                class_weight="balanced",
                random_state=seed,
            )
    else:
        raise ValueError("learner must be one of {}.".format(LEARNERS))


def fit_cell(
    learner: str,
    train: WindowDataset,
    validation: WindowDataset,
    columns: np.ndarray,
    seed: int,
    n_estimators: int,
    model_jobs: int,
    logistic_c_grid: Sequence[float] = LOGISTIC_C_GRID,
):
    """Select one setting on validation BA; return ``(model, label, value, val_ba, val_probs)``."""

    best = None
    limits = None if model_jobs == -1 else model_jobs
    for label, value, candidate in learner_candidates(
        learner, seed, n_estimators, model_jobs, logistic_c_grid
    ):
        # NumPy 2 linked against Accelerate emits spurious floating-point
        # warnings from matmul on finite float64 operands; the probabilities
        # are validated in score_cell instead.
        with threadpool_limits(limits=limits), np.errstate(
            divide="ignore", over="ignore", invalid="ignore"
        ):
            candidate.fit(train.X[:, columns], train.y)
            validation_probabilities = ordered_probabilities(candidate, validation.X[:, columns])
        score = float(
            balanced_accuracy_score(validation.y, np.argmax(validation_probabilities, axis=1))
        )
        if best is None or score > best[3]:
            best = (candidate, label, value, score, validation_probabilities)
    return best


def score_cell(
    splits: PathLevelSplits,
    learner: str,
    cell: str,
    columns: np.ndarray,
    seed: int,
    n_estimators: int,
    model_jobs: int,
    logistic_c_grid: Sequence[float] = LOGISTIC_C_GRID,
) -> Tuple[Dict[str, float], str, np.ndarray, np.ndarray]:
    """Fit one learner on one cell; return metrics, setting, and probabilities."""

    model, label, value, validation_ba, validation_probabilities = fit_cell(
        learner,
        splits.train,
        splits.validation,
        columns,
        seed,
        n_estimators,
        model_jobs,
        logistic_c_grid,
    )
    with threadpool_limits(limits=None if model_jobs == -1 else model_jobs), np.errstate(
        divide="ignore", over="ignore", invalid="ignore"
    ):
        test_probabilities = ordered_probabilities(model, splits.test.X[:, columns])
    if not (np.isfinite(validation_probabilities).all() and np.isfinite(test_probabilities).all()):
        raise FloatingPointError(
            "{} returned non-finite probabilities for {}.".format(learner, cell)
        )
    prefix = "{}_{}".format(learner, cell)
    row: Dict[str, float] = {
        prefix + "_n_features": int(len(columns)),
        prefix + "_selected_value": float(value),
        prefix + "_validation_ba": float(validation_ba),
    }
    row.update(
        classification_diagnostics(
            splits.test.y, np.argmax(test_probabilities, axis=1), prefix + "_test"
        )
    )
    return row, label, validation_probabilities, test_probabilities


def contrasts(row: Dict[str, float], learners: Iterable[str], cells: Iterable[str]) -> Dict[str, float]:
    """Paired representation contrasts within learners and learner contrasts within cells."""

    cells = tuple(cells)
    learners = tuple(learners)
    result: Dict[str, float] = {}
    for learner in learners:
        for minuend, subtrahend in CONTRASTS:
            if minuend in cells and subtrahend in cells:
                result["{}_{}_minus_{}_ba".format(learner, minuend, subtrahend)] = (
                    row["{}_{}_test_ba".format(learner, minuend)]
                    - row["{}_{}_test_ba".format(learner, subtrahend)]
                )
    for cell in cells:
        for first, second in LEARNER_CONTRASTS:
            if first in learners and second in learners:
                result["{}_{}_minus_{}_ba".format(cell, first, second)] = (
                    row["{}_{}_test_ba".format(first, cell)]
                    - row["{}_{}_test_ba".format(second, cell)]
                )
    return result


# ---------------------------------------------------------------------------
# Replication
# ---------------------------------------------------------------------------


def quantize(probabilities: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(probabilities, 0.0, 1.0) * PROBABILITY_QUANTUM).astype(np.uint16)


def dequantize(values: np.ndarray) -> np.ndarray:
    return values.astype(float) / PROBABILITY_QUANTUM


def save_probabilities(
    path: Path,
    splits: PathLevelSplits,
    store: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]],
) -> None:
    """Save uint16-quantized validation/test probabilities of the stored cells."""

    arrays: Dict[str, np.ndarray] = {}
    for split_name in ("validation", "test"):
        dataset = getattr(splits, split_name)
        arrays[split_name + "_y"] = dataset.y
        arrays[split_name + "_path_ids"] = dataset.path_ids.astype(str)
        arrays[split_name + "_end_times"] = dataset.end_times
    for (learner, cell), (validation_probabilities, test_probabilities) in store.items():
        arrays["validation__{}__{}".format(learner, cell)] = quantize(validation_probabilities)
        arrays["test__{}__{}".format(learner, cell)] = quantize(test_probabilities)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def save_paths(path: Path, splits: PathLevelSplits) -> None:
    """Save log returns and regime labels of every simulated path."""

    arrays: Dict[str, np.ndarray] = {}
    for split_name in ("train", "validation", "test"):
        for simulated in splits.paths[split_name]:
            key = "{}__{}".format(split_name, simulated.path_id)
            arrays[key + "__log_return"] = simulated.frame["LogReturn"].to_numpy(dtype=float)
            arrays[key + "__regime"] = simulated.frame["Regime"].to_numpy(dtype=np.int8)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def evaluate_replication(
    design: str,
    replication: int,
    seed: int,
    n_paths: Tuple[int, int, int],
    n_steps: int,
    window_size: int,
    p_switch: float,
    n_estimators: int,
    model_jobs: int = -1,
    learners: Tuple[str, ...] = LEARNERS,
    cells: Tuple[str, ...] = tuple(CELLS),
    output_dir: Optional[Path] = None,
    store_cells: Tuple[str, ...] = TABLE_CELLS,
    logistic_c_grid: Sequence[float] = LOGISTIC_C_GRID,
) -> Dict[str, float]:
    """Simulate once, fit every learner on every cell, and score each on test."""

    for cell in cells:
        if cell not in CELLS:
            raise ValueError("Unknown cell: {}".format(cell))
    for learner in learners:
        if learner not in LEARNERS:
            raise ValueError("Unknown learner: {}".format(learner))
    splits = build_grid_splits(design, seed, n_paths, n_steps, window_size, p_switch)
    # The filter costs five EM restarts per replication, so it is fitted only
    # when a requested cell uses it; the paths and every other block are
    # unaffected, so the remaining cells reproduce a full run exactly.
    hmm_diagnostics: Dict[str, float] = {}
    if "hmm" in cells:
        splits, hmm_diagnostics = add_hmm_block(splits, seed, window_size)
    selections = block_columns(splits.train.feature_names)
    columns_by_cell = cell_columns(selections, cells)

    row: Dict[str, float] = {
        "design": design,
        "replication": int(replication),
        "seed": int(seed),
        "n_train_windows": int(len(splits.train)),
        "n_validation_windows": int(len(splits.validation)),
        "n_test_windows": int(len(splits.test)),
    }
    row.update(split_class_counts(splits))
    row.update(hmm_diagnostics)
    store: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    for learner in learners:
        for cell in cells:
            metrics, label, validation_probabilities, test_probabilities = score_cell(
                splits,
                learner,
                cell,
                columns_by_cell[cell],
                seed,
                n_estimators,
                model_jobs,
                logistic_c_grid,
            )
            row.update(metrics)
            row["{}_{}_selected_setting".format(learner, cell)] = label
            if output_dir is not None and cell in store_cells:
                store[(learner, cell)] = (validation_probabilities, test_probabilities)
    row.update(contrasts(row, learners, cells))
    if output_dir is not None:
        stem = "replication_{:02d}.npz".format(replication)
        save_probabilities(Path(output_dir) / "probabilities" / stem, splits, store)
        save_paths(Path(output_dir) / "paths" / stem, splits)
    return row


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def cell_table(results: pd.DataFrame) -> pd.DataFrame:
    """Tidy one-row-per-(replication, learner, cell) view of the wide results."""

    rows = []
    for _, wide in results.iterrows():
        for learner in LEARNERS:
            for cell in CELLS:
                prefix = "{}_{}".format(learner, cell)
                if prefix + "_test_ba" not in wide or pd.isna(wide[prefix + "_test_ba"]):
                    continue
                record = {
                    "design": wide["design"],
                    "replication": int(wide["replication"]),
                    "seed": int(wide["seed"]),
                    "learner": learner,
                    "cell": cell,
                    "n_features": int(wide[prefix + "_n_features"]),
                    "selected_setting": wide[prefix + "_selected_setting"],
                    "validation_ba": wide[prefix + "_validation_ba"],
                    "test_ba": wide[prefix + "_test_ba"],
                    "test_macro_f1": wide[prefix + "_test_macro_f1"],
                    "test_accuracy": wide[prefix + "_test_accuracy"],
                }
                for regime in REGIME_IDS:
                    record["test_recall_{}".format(regime)] = wide[
                        prefix + "_test_recall_{}".format(regime)
                    ]
                rows.append(record)
    return pd.DataFrame(rows)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Levels and paired contrasts with Student-t intervals and positive counts."""

    rows = []

    def add(label: str, column: str, count_positive: bool) -> None:
        if column not in results.columns:
            return
        values = results[column].dropna()
        if len(values) < 2:
            return
        record = {"metric": label, **mean_t_interval(values)}
        if count_positive:
            record["positive_count"] = int(np.sum(values > 0))
            record["positive_share"] = float(np.mean(values > 0))
        rows.append(record)

    add("BA_hmm_direct", "hmm_direct_test_ba", False)
    for learner in LEARNERS:
        for cell in CELLS:
            add("BA_{}_{}".format(learner, cell), "{}_{}_test_ba".format(learner, cell), False)
    for learner in LEARNERS:
        for minuend, subtrahend in CONTRASTS:
            add(
                "Delta_{}_{}_minus_{}".format(learner, minuend, subtrahend),
                "{}_{}_minus_{}_ba".format(learner, minuend, subtrahend),
                True,
            )
    for cell in CELLS:
        for first, second in LEARNER_CONTRASTS:
            add(
                "Delta_{}_{}_minus_{}".format(cell, first, second),
                "{}_{}_minus_{}_ba".format(cell, first, second),
                True,
            )
    for label in REGIME_IDS:
        add("HMM_state_std_{}".format(label), "hmm_state_std_{}".format(label), False)
        add("HMM_persistence_{}".format(label), "hmm_persistence_{}".format(label), False)
    add("HMM_log_likelihood", "hmm_log_likelihood", False)
    add("HMM_selected_restart", "hmm_selected_restart", False)
    return pd.DataFrame(rows)


def save_tables(results: pd.DataFrame, output_dir: Path) -> None:
    results.to_csv(output_dir / "replications.csv", index=False)
    cell_table(results).to_csv(output_dir / "cells.csv", index=False)
    if len(results) >= 2:
        summarize_results(results).to_csv(output_dir / "summary.csv", index=False)


def progress_line(row: Dict[str, float], learners: Iterable[str], cells: Iterable[str]) -> str:
    cells = set(cells)
    parts = []
    for learner in learners:
        def ba(cell: str) -> str:
            return "{:.4f}".format(row["{}_{}_test_ba".format(learner, cell)]) if cell in cells else "-"

        def gain(minuend: str, subtrahend: str) -> str:
            key = "{}_{}_minus_{}_ba".format(learner, minuend, subtrahend)
            return "{:+.4f}".format(row[key]) if key in row else "-"

        parts.append(
            "{} stat={} (+log {} +raw {}) b3={} (+log {} +raw {}) hmm={}".format(
                LEARNER_SHORT[learner],
                ba("statistics"),
                gain("statistics_logsig", "statistics"),
                gain("statistics_rawsig", "statistics"),
                ba("b3"),
                gain("b3_logsig", "b3"),
                gain("b3_rawsig", "b3"),
                ba("hmm"),
            )
        )
    if "hmm_direct_test_ba" in row:
        parts.append("HMM direct={:.4f}".format(row["hmm_direct_test_ba"]))
    return "\n  ".join(parts)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _evaluate_task(task: Dict[str, object]) -> Dict[str, float]:
    started = time.monotonic()
    row = evaluate_replication(**task)
    row["elapsed_seconds"] = float(time.monotonic() - started)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", choices=tuple(DESIGNS), required=True)
    parser.add_argument("--replications", type=int, default=50)
    parser.add_argument("--train-paths", type=int, default=10)
    parser.add_argument("--validation-paths", type=int, default=10)
    parser.add_argument("--test-paths", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--p-switch", type=float, default=0.002)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument(
        "--logistic-c-grid",
        nargs="+",
        type=float,
        default=list(LOGISTIC_C_GRID),
        help="Inverse l2 penalties searched for logistic regression (default: the main grid's).",
    )
    parser.add_argument("--learners", nargs="+", choices=LEARNERS, default=list(LEARNERS))
    parser.add_argument("--cells", nargs="+", choices=tuple(CELLS), default=list(CELLS))
    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help="Defaults to the main-grid row's seed so replications are paired with it.",
    )
    parser.add_argument("--jobs", type=int, default=1, help="Parallel replications.")
    parser.add_argument(
        "--no-save-probabilities",
        action="store_true",
        help="Skip the per-replication probability and path archives.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replications < 2:
        raise ValueError("At least two replications are required.")
    if min(args.train_paths, args.validation_paths, args.test_paths) < 1 or args.jobs < 1:
        raise ValueError("Every split needs at least one path and jobs must be positive.")
    base_seed = DESIGNS[args.design]["base_seed"] if args.base_seed is None else args.base_seed
    output_dir = (
        Path("results/unified_grid") / args.design if args.output_dir is None else args.output_dir
    ).resolve()
    guard_output_directory(output_dir, args.resume)
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = tuple(cell for cell in CELLS if cell in set(args.cells))
    learners = tuple(learner for learner in LEARNERS if learner in set(args.learners))
    config = {
        "design": args.design,
        "replications": args.replications,
        "n_paths": [args.train_paths, args.validation_paths, args.test_paths],
        "n_steps": args.steps,
        "window_size": args.window,
        "p_switch": args.p_switch,
        "simulation_substeps": DESIGNS[args.design]["substeps"],
        "stationary_initial_variance": True,
        "base_seed": base_seed,
        "signature_level": 3,
        "signature_path": CausalFeatureExtractor.PATH_CONVENTION,
        "statistics_columns": list(STATISTICS_COLUMNS),
        "cells": {cell: list(CELLS[cell]) for cell in cells},
        "table_cells": list(TABLE_CELLS),
        "learners": list(learners),
        "n_estimators": args.trees,
        "rf_depth_grid": list(RF_DEPTH_GRID),
        "logistic_c_grid": list(args.logistic_c_grid),
        "gbm_leaf_grid": list(GBM_LEAF_GRID),
        "gbm_learning_rate": GBM_LEARNING_RATE,
        "gbm_max_iter": GBM_MAX_ITER,
        "hmm": {
            "states": HMM_STATES,
            "return_scale": HMM_RETURN_SCALE,
            "min_covar": HMM_MIN_COVAR,
            "max_iter": HMM_MAX_ITER,
            "restarts": HMM_RESTARTS,
            "persistent_initialization_diagonal": HMM_PERSISTENT_DIAGONAL,
            "selection": "highest training log-likelihood",
            "matching": "training-window argmax agreement (Hungarian)",
        },
        "save_probabilities": not args.no_save_probabilities,
    }
    config_path = output_dir / "config.json"
    if args.resume and config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Cannot resume: current arguments differ from config.json.")
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    replications_path = output_dir / "replications.csv"
    if args.resume and replications_path.exists():
        existing = pd.read_csv(replications_path)
        completed = set(existing["replication"].astype(int).tolist())
        rows = existing.to_dict("records")
    else:
        completed = set()
        rows = []

    tasks = []
    for replication, seed in enumerate(replication_seeds(base_seed, args.replications), start=1):
        if replication in completed:
            continue
        tasks.append(
            {
                "design": args.design,
                "replication": replication,
                "seed": seed,
                "n_paths": (args.train_paths, args.validation_paths, args.test_paths),
                "n_steps": args.steps,
                "window_size": args.window,
                "p_switch": args.p_switch,
                "n_estimators": args.trees,
                "model_jobs": -1 if args.jobs == 1 else 1,
                "learners": learners,
                "cells": cells,
                "output_dir": None if args.no_save_probabilities else output_dir,
                "logistic_c_grid": tuple(args.logistic_c_grid),
            }
        )

    def checkpoint(row: Dict[str, float]) -> None:
        rows.append(row)
        save_tables(pd.DataFrame(rows).sort_values("replication").reset_index(drop=True), output_dir)
        print(
            "Replication {}/{} seed={} ({:.0f}s)\n  {}".format(
                int(row["replication"]),
                args.replications,
                int(row["seed"]),
                row["elapsed_seconds"],
                progress_line(row, learners, cells),
            ),
            flush=True,
        )

    started = time.monotonic()
    if args.jobs == 1:
        for task in tasks:
            checkpoint(_evaluate_task(task))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(_evaluate_task, task) for task in tasks]
            for future in as_completed(futures):
                checkpoint(future.result())

    results = pd.DataFrame(rows).sort_values("replication").reset_index(drop=True)
    save_tables(results, output_dir)
    print("\nSummary ({} replications)".format(len(results)))
    print(
        summarize_results(results).to_string(
            index=False, float_format=lambda value: "{:.6f}".format(value)
        )
    )
    print("\nSaved to {}".format(output_dir))
    print("Total time: {:.1f}s".format(time.monotonic() - started))


if __name__ == "__main__":
    main()
