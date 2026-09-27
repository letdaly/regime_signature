"""Paired Monte Carlo study for Experiment A (pure variance-scale shifts).

For every replication, paths and combined features are generated exactly once.
Statistics, Signature, and Combined Random Forests then receive column subsets
of that same dataset.  The resulting balanced accuracies are paired by seed.
A validation-tuned multinomial logistic regression is fitted to the same three
column subsets as a second learner (``--learners``), because the mechanism
benchmark showed that a linear model extracts a larger signature increment
than the forest in the matched-marginal design.

Every representation is also passed through the causal transition detector
shared with Experiment B (``regime_detection``): smoothing, confidence, and
confirmation settings are selected on validation paths only, and detection
delay, false-switch rate, and boundary F1 are then measured once on test
paths.  This gives the standard-Heston rows of the main table the same metric
set as the matched-marginal row.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler

from regime_detection import (
    DETECTOR_CONTRAST_METRICS,
    DETECTOR_GRID,
    REGIME_IDS,
    TIME_SINCE_CHANGE_BUCKETS,
    detector_metrics,
    ordered_probabilities,
    stratified_classification,
    time_since_last_change,
    tune_detector,
)
from regime_pipeline import (
    CausalFeatureExtractor,
    DEFAULT_N_PATHS,
    PathLevelSplits,
    build_path_level_splits,
    sample_scale_only_parameters,
)


REPRESENTATIONS = ("statistics", "signature", "combined")
DEFAULT_EVENT_TOLERANCE = 100
# Log-signatures are the main-grid representation (14 coordinates at level 3
# on the three-channel path, versus 39 raw coordinates).  "raw" reproduces the
# frozen (10, 3, 5) studies, which predate the iisignature build that can
# prepare log-signatures on this platform.
DEFAULT_SIGNATURE_KIND = "log"
SIGNATURE_KINDS = ("log", "raw")
# The Random Forest is the fixed main-table learner and keeps the bare
# representation names in every output (``statistics_test_ba``), so its
# columns line up with the frozen studies.  The logistic learner's columns are
# prefixed (``logistic_statistics_test_ba``).
LEARNERS = ("random_forest", "logistic")
DEFAULT_LEARNERS = LEARNERS
LOGISTIC_C_GRID = (0.01, 0.1, 1.0)


def result_prefix(learner: str, representation: str) -> str:
    """Column prefix of one learner-representation cell of the main table."""

    if learner not in LEARNERS:
        raise ValueError("learner must be one of {}.".format(LEARNERS))
    if learner == "random_forest":
        return representation
    return learner + "_" + representation


def clip_scaled_features(values: np.ndarray) -> np.ndarray:
    """Bound training-scaled tails to keep the logistic optimization well conditioned."""

    return np.clip(values, -20.0, 20.0)


def fit_learner(
    learner: str,
    train_X: np.ndarray,
    train_y: np.ndarray,
    validation_X: np.ndarray,
    validation_y: np.ndarray,
    seed: int,
    n_estimators: int,
    model_jobs: int,
):
    """Fit one learner on training windows; return ``(model, selected_c)``.

    The forest is the fixed classifier of the plan (no tuning).  The logistic
    learner robust-scales features on training windows only, then selects its
    regularization strength from ``LOGISTIC_C_GRID`` by validation balanced
    accuracy, as the mechanism benchmark does; test windows are never seen
    before scoring.  ``selected_c`` is NaN for the forest.
    """

    if learner == "random_forest":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            class_weight="balanced",
            random_state=seed,
            n_jobs=model_jobs,
        )
        model.fit(train_X, train_y)
        return model, float("nan")
    if learner != "logistic":
        raise ValueError("learner must be one of {}.".format(LEARNERS))
    best = None
    for c_value in LOGISTIC_C_GRID:
        candidate = Pipeline(
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
        # NumPy 2 linked against Accelerate emits spurious floating-point
        # warnings from matmul on finite float64 operands; the probabilities
        # are validated in score_cell instead.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            candidate.fit(train_X, train_y)
            validation_prediction = candidate.predict(validation_X)
        score = float(balanced_accuracy_score(validation_y, validation_prediction))
        if best is None or score > best[0]:
            best = (score, c_value, candidate)
    return best[2], float(best[1])


def split_class_counts(splits: PathLevelSplits) -> Dict[str, int]:
    """Count endpoint-labeled windows by regime in every data partition."""

    counts: Dict[str, int] = {}
    for split_name in ("train", "validation", "test"):
        labels = getattr(splits, split_name).y
        for regime in REGIME_IDS:
            counts["{}_n{}".format(split_name, regime)] = int(np.sum(labels == regime))
    return counts


def classification_diagnostics(
    y_true: np.ndarray, y_pred: np.ndarray, prefix: str
) -> Dict[str, float]:
    """Return fixed-label recall, BA, macro F1, accuracy, and 3x3 confusion cells."""

    recalls = recall_score(
        y_true, y_pred, labels=REGIME_IDS, average=None, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=REGIME_IDS)
    diagnostics: Dict[str, float] = {
        prefix + "_accuracy": float(accuracy_score(y_true, y_pred)),
        prefix + "_ba": float(balanced_accuracy_score(y_true, y_pred)),
        prefix + "_macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=REGIME_IDS,
                average="macro",
                zero_division=0,
            )
        ),
    }
    for regime, recall in zip(REGIME_IDS, recalls):
        diagnostics[prefix + "_recall_{}".format(regime)] = float(recall)
    for true_regime in REGIME_IDS:
        for predicted_regime in REGIME_IDS:
            diagnostics[
                prefix + "_cm_{}_{}".format(true_regime, predicted_regime)
            ] = int(matrix[true_regime, predicted_regime])
    return diagnostics


def feature_indices(feature_names: Iterable[str]) -> Dict[str, np.ndarray]:
    """Return deterministic column selections from one combined feature matrix."""

    names = tuple(feature_names)
    def is_signature(name: str) -> bool:
        return name.startswith("signature_") or name.startswith("logsignature_")

    statistics = np.array(
        [index for index, name in enumerate(names) if not is_signature(name)], dtype=int
    )
    signature = np.array(
        [index for index, name in enumerate(names) if is_signature(name)], dtype=int
    )
    if len(statistics) == 0 or len(signature) == 0:
        raise ValueError("Combined data must contain statistics and signature columns.")
    return {
        "statistics": statistics,
        "signature": signature,
        "combined": np.arange(len(names), dtype=int),
    }


def evaluate_paired_replication(
    replication: int,
    seed: int,
    n_paths: Tuple[int, int, int],
    n_steps: int,
    window_size: int,
    p_switch: float,
    n_estimators: int,
    model_jobs: int = -1,
    event_tolerance: int = DEFAULT_EVENT_TOLERANCE,
    evaluate_detector: bool = True,
    signature_kind: str = DEFAULT_SIGNATURE_KIND,
    learners: Tuple[str, ...] = DEFAULT_LEARNERS,
) -> Dict[str, float]:
    """Evaluate all representations on one shared simulated path collection."""

    extractor = CausalFeatureExtractor(
        window_size=window_size,
        short_window=min(10, window_size),
        signature_level=3,
        representation="combined",
        signature_kind=signature_kind,
    )
    splits = build_path_level_splits(
        extractor=extractor,
        parameter_sampler=sample_scale_only_parameters,
        n_paths=n_paths,
        n_steps=n_steps,
        master_seed=seed,
        p_switch=p_switch,
        min_dwell=window_size,
        stationary_initial_variance=True,
    )
    return evaluate_splits(
        splits,
        replication=replication,
        seed=seed,
        n_estimators=n_estimators,
        model_jobs=model_jobs,
        event_tolerance=event_tolerance,
        evaluate_detector=evaluate_detector,
        learners=learners,
    )


def evaluate_splits(
    splits: PathLevelSplits,
    replication: int,
    seed: int,
    n_estimators: int,
    model_jobs: int = -1,
    event_tolerance: int = DEFAULT_EVENT_TOLERANCE,
    evaluate_detector: bool = True,
    learners: Tuple[str, ...] = DEFAULT_LEARNERS,
) -> Dict[str, float]:
    """Fit every learner-representation cell on prepared splits and score it.

    Shared by the Experiment A and fBM-driven rows so that every cell of the
    main table is produced by literally the same classifier, detector, and
    metric code; only the simulated paths differ.
    """

    splits.assert_disjoint()
    selections = feature_indices(splits.train.feature_names)

    row: Dict[str, float] = {
        "replication": int(replication),
        "seed": int(seed),
        "n_train_windows": int(len(splits.train)),
        "n_validation_windows": int(len(splits.validation)),
        "n_test_windows": int(len(splits.test)),
    }
    row.update(split_class_counts(splits))
    test_distances = time_since_last_change(splits.test, splits.paths["test"])

    for learner in learners:
        for representation in REPRESENTATIONS:
            columns = selections[representation]
            row.update(
                score_cell(
                    splits,
                    learner,
                    representation,
                    columns,
                    test_distances,
                    seed=seed,
                    n_estimators=n_estimators,
                    model_jobs=model_jobs,
                    event_tolerance=event_tolerance,
                    evaluate_detector=evaluate_detector,
                )
            )
        row.update(representation_contrasts(row, learner, evaluate_detector))
    row.update(learner_contrasts(row, learners))
    return row


def score_cell(
    splits: PathLevelSplits,
    learner: str,
    representation: str,
    columns: np.ndarray,
    test_distances: np.ndarray,
    seed: int,
    n_estimators: int,
    model_jobs: int,
    event_tolerance: int,
    evaluate_detector: bool,
    extra_diagnostics: Optional[
        Callable[[np.ndarray, np.ndarray, np.ndarray, str], Dict[str, float]]
    ] = None,
) -> Dict[str, float]:
    """Fit one learner on one column subset and score classification and detection.

    ``extra_diagnostics(y_true, y_pred, probabilities, prefix)`` lets a runner
    add probability-based test diagnostics (Experiment B's classwise metrics)
    without repeating the fitting and detector code.
    """

    model, selected_c = fit_learner(
        learner,
        splits.train.X[:, columns],
        splits.train.y,
        splits.validation.X[:, columns],
        splits.validation.y,
        seed=seed,
        n_estimators=n_estimators,
        model_jobs=model_jobs,
    )
    # argmax over fixed-order probabilities equals model.predict for both
    # learners, so the classification metrics are unchanged; the
    # probabilities themselves feed the causal detector.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        validation_probabilities = ordered_probabilities(
            model, splits.validation.X[:, columns]
        )
        test_probabilities = ordered_probabilities(model, splits.test.X[:, columns])
    if not (np.isfinite(validation_probabilities).all() and np.isfinite(test_probabilities).all()):
        raise FloatingPointError(
            "{} returned non-finite probabilities for {}.".format(learner, representation)
        )
    validation_prediction = np.argmax(validation_probabilities, axis=1)
    test_prediction = np.argmax(test_probabilities, axis=1)

    prefix = result_prefix(learner, representation)
    row: Dict[str, float] = {
        prefix + "_n_features": int(len(columns)),
        prefix + "_validation_ba": float(
            balanced_accuracy_score(splits.validation.y, validation_prediction)
        ),
    }
    if learner == "logistic":
        row[prefix + "_selected_c"] = selected_c
    row.update(
        classification_diagnostics(splits.test.y, test_prediction, prefix + "_test")
    )
    row.update(
        stratified_classification(
            splits.test.y, test_prediction, test_distances, prefix + "_test"
        )
    )
    if extra_diagnostics is not None:
        row.update(
            extra_diagnostics(splits.test.y, test_prediction, test_probabilities, prefix + "_test")
        )
    if evaluate_detector:
        detector_parameters, validation_detector = tune_detector(
            splits.validation,
            validation_probabilities,
            tolerance=event_tolerance,
        )
        alpha, threshold, confirmations = detector_parameters
        row[prefix + "_detector_alpha"] = alpha
        row[prefix + "_detector_threshold"] = threshold
        row[prefix + "_detector_confirmations"] = confirmations
        row[prefix + "_validation_boundary_f1"] = validation_detector["boundary_f1"]
        test_detector = detector_metrics(
            splits.test,
            test_probabilities,
            alpha=alpha,
            threshold=threshold,
            confirmations=confirmations,
            tolerance=event_tolerance,
        )
        for name, value in test_detector.items():
            row[prefix + "_test_" + name] = value
    return row


def representation_contrasts(
    row: Dict[str, float], learner: str, evaluate_detector: bool
) -> Dict[str, float]:
    """Paired within-learner contrasts, e.g. ``combined_minus_statistics_ba``."""

    def cell(representation: str) -> str:
        return result_prefix(learner, representation)

    contrasts: Dict[str, float] = {
        cell("combined") + "_minus_statistics_ba": (
            row[cell("combined") + "_test_ba"] - row[cell("statistics") + "_test_ba"]
        ),
        cell("signature") + "_minus_statistics_ba": (
            row[cell("signature") + "_test_ba"] - row[cell("statistics") + "_test_ba"]
        ),
        cell("combined") + "_minus_signature_ba": (
            row[cell("combined") + "_test_ba"] - row[cell("signature") + "_test_ba"]
        ),
    }
    for candidate in ("combined", "signature"):
        for name, _, _ in TIME_SINCE_CHANGE_BUCKETS:
            contrasts[cell(candidate) + "_minus_statistics_ba_since_" + name] = (
                row[cell(candidate) + "_test_ba_since_" + name]
                - row[cell("statistics") + "_test_ba_since_" + name]
            )
        if evaluate_detector:
            for metric in DETECTOR_CONTRAST_METRICS:
                contrasts[cell(candidate) + "_minus_statistics_" + metric] = (
                    row[cell(candidate) + "_test_" + metric]
                    - row[cell("statistics") + "_test_" + metric]
                )
    return contrasts


def learner_contrasts(row: Dict[str, float], learners: Tuple[str, ...]) -> Dict[str, float]:
    """Paired logistic-minus-forest balanced accuracy for each representation."""

    if "random_forest" not in learners or "logistic" not in learners:
        return {}
    return {
        "logistic_minus_random_forest_{}_ba".format(representation): (
            row[result_prefix("logistic", representation) + "_test_ba"]
            - row[result_prefix("random_forest", representation) + "_test_ba"]
        )
        for representation in REPRESENTATIONS
    }


def mean_t_interval(values: Iterable[float], confidence: float = 0.95) -> Dict[str, float]:
    """Mean and two-sided Student-t confidence interval across replications."""

    array = np.asarray(tuple(values), dtype=float)
    if len(array) < 2:
        raise ValueError("At least two independent replications are required.")
    mean = float(np.mean(array))
    standard_deviation = float(np.std(array, ddof=1))
    standard_error = standard_deviation / np.sqrt(len(array))
    critical_value = float(t.ppf(0.5 + confidence / 2.0, df=len(array) - 1))
    half_width = critical_value * standard_error
    return {
        "n": int(len(array)),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": float(standard_error),
        "confidence": float(confidence),
        "ci_low": float(mean - half_width),
        "ci_high": float(mean + half_width),
    }


DETECTOR_SUMMARY_LABELS = {
    "detector_balanced_accuracy": "Detector_BA",
    "boundary_f1": "Boundary_F1",
    "detection_probability": "Detection_probability",
    "false_switches_per_1000": "False_switches_per_1000",
    "mean_detection_delay": "Mean_delay",
}


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize balanced accuracies, detector metrics, and paired differences.

    Detector rows are included only when their columns exist, so summaries of
    classification-only runs (including the frozen scale-only results) still
    work.  Delays are NaN for replications without any matched change and are
    dropped before averaging, exactly as in Experiment B.
    """

    metrics = {}
    # Forest rows keep their historical labels ("BA_statistics"); the logistic
    # learner's rows carry its prefix ("BA_logistic_statistics").
    for learner in LEARNERS:
        cells = {
            representation: result_prefix(learner, representation)
            for representation in REPRESENTATIONS
        }
        for representation, cell in cells.items():
            metrics["BA_" + cell] = cell + "_test_ba"
        metrics["Delta_" + cells["combined"] + "_minus_statistics"] = (
            cells["combined"] + "_minus_statistics_ba"
        )
        metrics["Delta_" + cells["signature"] + "_minus_statistics"] = (
            cells["signature"] + "_minus_statistics_ba"
        )
        metrics["Delta_" + cells["combined"] + "_minus_signature"] = (
            cells["combined"] + "_minus_signature_ba"
        )
        for name, _, _ in TIME_SINCE_CHANGE_BUCKETS:
            for representation, cell in cells.items():
                metrics["BA_since_{}_{}".format(name, cell)] = cell + "_test_ba_since_" + name
            for candidate in ("combined", "signature"):
                metrics["Delta_BA_since_{}_{}_minus_statistics".format(name, cells[candidate])] = (
                    cells[candidate] + "_minus_statistics_ba_since_" + name
                )
        for metric, label in DETECTOR_SUMMARY_LABELS.items():
            for representation, cell in cells.items():
                metrics[label + "_" + cell] = cell + "_test_" + metric
            for candidate in ("combined", "signature"):
                metrics["Delta_" + label + "_" + cells[candidate] + "_minus_statistics"] = (
                    cells[candidate] + "_minus_statistics_" + metric
                )
    for representation in REPRESENTATIONS:
        metrics["Delta_BA_logistic_minus_random_forest_" + representation] = (
            "logistic_minus_random_forest_{}_ba".format(representation)
        )
    rows = []
    for label, column in metrics.items():
        if column not in results.columns:
            continue
        values = results[column].dropna()
        if len(values) >= 2:
            rows.append({"metric": label, **mean_t_interval(values)})
    return pd.DataFrame(rows)


def diagnostic_tables(results: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create tidy class-count and per-method test diagnostic tables."""

    count_rows = []
    diagnostic_rows = []
    for _, row in results.iterrows():
        identity = {
            "replication": int(row["replication"]),
            "seed": int(row["seed"]),
        }
        for split_name in ("train", "validation", "test"):
            count_rows.append(
                {
                    **identity,
                    "split": split_name,
                    **{
                        "n{}".format(regime): int(
                            row["{}_n{}".format(split_name, regime)]
                        )
                        for regime in REGIME_IDS
                    },
                }
            )
        for learner in LEARNERS:
            for representation in REPRESENTATIONS:
                cell = result_prefix(learner, representation)
                if cell + "_test_ba" not in row:
                    continue
                diagnostic_rows.append(
                    cell_diagnostics(row, identity, learner, representation, cell)
                )
    return pd.DataFrame(count_rows), pd.DataFrame(diagnostic_rows)


def cell_diagnostics(
    row: pd.Series, identity: Dict[str, int], learner: str, representation: str, cell: str
) -> Dict[str, float]:
    """One tidy test-diagnostics record for a learner-representation cell."""

    prefix = cell + "_test"
    diagnostic = {
        **identity,
        "learner": learner,
        "representation": representation,
        "balanced_accuracy": row[prefix + "_ba"],
        "macro_f1": row[prefix + "_macro_f1"],
    }
    if learner == "logistic":
        diagnostic["selected_c"] = row[cell + "_selected_c"]
    for regime in REGIME_IDS:
        diagnostic["recall_{}".format(regime)] = row[prefix + "_recall_{}".format(regime)]
    for true_regime in REGIME_IDS:
        for predicted_regime in REGIME_IDS:
            diagnostic["cm_{}_{}".format(true_regime, predicted_regime)] = int(
                row[prefix + "_cm_{}_{}".format(true_regime, predicted_regime)]
            )
    for name, _, _ in TIME_SINCE_CHANGE_BUCKETS:
        if prefix + "_ba_since_" + name in row:
            diagnostic["ba_since_" + name] = row[prefix + "_ba_since_" + name]
            diagnostic["n_since_" + name] = int(row[prefix + "_n_since_" + name])
    if prefix + "_boundary_f1" in row:
        for name in (
            "detector_balanced_accuracy",
            "detector_macro_f1",
            "fraction_time_wrong",
            "brier_score",
            "detection_probability",
            "mean_detection_delay",
            "median_detection_delay",
            "false_switches_per_1000",
            "boundary_precision",
            "boundary_recall",
            "boundary_f1",
            "n_true_changes",
            "n_matched_changes",
            "n_false_switches",
        ):
            diagnostic[name] = row[prefix + "_" + name]
        diagnostic["alpha"] = row[cell + "_detector_alpha"]
        diagnostic["threshold"] = row[cell + "_detector_threshold"]
        diagnostic["confirmations"] = int(row[cell + "_detector_confirmations"])
    return diagnostic


def save_result_tables(
    results: pd.DataFrame,
    replications_path: Path,
    summary_path: Path,
    class_counts_path: Path,
    diagnostics_path: Path,
) -> None:
    """Checkpoint all wide and tidy experiment outputs."""

    results.to_csv(replications_path, index=False)
    class_counts, diagnostics = diagnostic_tables(results)
    class_counts.to_csv(class_counts_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    if len(results) >= 2:
        summarize_results(results).to_csv(summary_path, index=False)


def guard_output_directory(output_dir: Path, resume: bool) -> None:
    """Refuse to start a fresh run on top of an existing replications.csv.

    Result directories are frozen once written.  Continuing an interrupted run
    requires ``--resume``; anything else must go to a new ``--output-dir``.
    """

    existing = output_dir / "replications.csv"
    if existing.exists() and not resume:
        raise FileExistsError(
            "{} already holds results. Pass --resume to continue that run or "
            "choose a new --output-dir; frozen directories are never overwritten.".format(
                output_dir
            )
        )


def add_signature_kind_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--signature-kind",
        choices=SIGNATURE_KINDS,
        default=DEFAULT_SIGNATURE_KIND,
        help="log (main grid, 14 coordinates) or raw (frozen studies, 39 coordinates).",
    )


def add_learner_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--learners",
        nargs="+",
        choices=LEARNERS,
        default=list(DEFAULT_LEARNERS),
        help="Learners fitted to every representation (default: both). The forest "
        "is the main-table classifier; logistic columns are prefixed logistic_.",
    )


def add_path_count_arguments(parser: argparse.ArgumentParser) -> None:
    """Shared --train-paths/--validation-paths/--test-paths with the grid defaults."""

    train, validation, test = DEFAULT_N_PATHS
    parser.add_argument("--train-paths", type=int, default=train)
    parser.add_argument(
        "--validation-paths",
        type=int,
        default=validation,
        help="Paths used only for detector/hyperparameter selection (default {}).".format(
            validation
        ),
    )
    parser.add_argument("--test-paths", type=int, default=test)


def replication_seeds(base_seed: int, replications: int) -> Tuple[int, ...]:
    """Create reproducible, independently spawned uint32 seeds."""

    children = np.random.SeedSequence(base_seed).spawn(replications)
    return tuple(int(child.generate_state(1, dtype=np.uint32)[0]) for child in children)


def progress_line(row: Dict[str, float], learners: Iterable[str]) -> str:
    """Balanced accuracies and Combined-minus-Statistics gain for each learner."""

    parts = []
    for learner in learners:
        cells = [result_prefix(learner, representation) for representation in REPRESENTATIONS]
        parts.append(
            "{} BA(stat/sig/comb)={:.4f}/{:.4f}/{:.4f} delta={:+.4f}".format(
                "RF" if learner == "random_forest" else "logit",
                *[row[cell + "_test_ba"] for cell in cells],
                row[cells[2] + "_minus_statistics_ba"],
            )
        )
    return " | ".join(parts)


def _evaluate_task(task: Dict[str, object]) -> Dict[str, float]:
    """Pickle-friendly worker entry point for process-level parallelism."""

    started = time.monotonic()
    row = evaluate_paired_replication(**task)
    row["elapsed_seconds"] = float(time.monotonic() - started)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=50)
    add_path_count_arguments(parser)
    add_signature_kind_argument(parser)
    add_learner_argument(parser)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--p-switch", type=float, default=0.002)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--event-tolerance", type=int, default=DEFAULT_EVENT_TOLERANCE)
    parser.add_argument(
        "--skip-detector",
        action="store_true",
        help="Report classification metrics only (no validation-tuned detector).",
    )
    parser.add_argument("--base-seed", type=int, default=20260827)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel replications. Forests use one thread when jobs > 1.",
    )
    # Main-table grid directory for the 10-validation-path configuration.  The
    # frozen 3-validation-path studies live in results/experiment_a_scale_only*
    # and are never written to.
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/main_grid/heston_scale_only"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run from the checkpoint CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replications < 2:
        raise ValueError("Experiment A requires at least two replications.")
    if min(args.train_paths, args.validation_paths, args.test_paths) < 1:
        raise ValueError("Every split needs at least one path.")
    if args.jobs < 1 or args.event_tolerance < 1:
        raise ValueError("jobs and event-tolerance must be positive.")

    output_dir = args.output_dir.resolve()
    guard_output_directory(output_dir, args.resume)
    output_dir.mkdir(parents=True, exist_ok=True)
    replications_path = output_dir / "replications.csv"
    summary_path = output_dir / "summary.csv"
    class_counts_path = output_dir / "class_counts.csv"
    diagnostics_path = output_dir / "test_diagnostics.csv"
    config_path = output_dir / "config.json"
    config = {
        "replications": args.replications,
        "n_paths": [args.train_paths, args.validation_paths, args.test_paths],
        "n_steps": args.steps,
        "window_size": args.window,
        "p_switch": args.p_switch,
        "n_estimators": args.trees,
        "base_seed": args.base_seed,
        "signature_kind": args.signature_kind,
        "signature_level": 3,
        "signature_path": CausalFeatureExtractor.PATH_CONVENTION,
        "learners": list(args.learners),
        "logistic_c_grid": list(LOGISTIC_C_GRID),
        "parameterization": "scale_only",
        "shared_parameters": ["mu", "kappa", "rho", "c"],
        "theta_ranges": {
            "0": [0.015, 0.025],
            "1": [0.065, 0.095],
            "2": [0.14, 0.20],
        },
        "c_range": [0.45, 0.55],
        "stationary_initial_variance": True,
        "evaluate_detector": not args.skip_detector,
        "event_tolerance": args.event_tolerance,
        "detector_grid": [list(values) for values in DETECTOR_GRID],
    }

    if args.resume and config_path.exists():
        existing_config = json.loads(config_path.read_text())
        if existing_config != config:
            raise ValueError("Cannot resume: current arguments differ from config.json.")
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    if args.resume and replications_path.exists():
        existing = pd.read_csv(replications_path)
        completed = set(existing["replication"].astype(int).tolist())
        rows = existing.to_dict("records")
    else:
        completed = set()
        rows = []

    seeds = replication_seeds(args.base_seed, args.replications)
    n_paths = (args.train_paths, args.validation_paths, args.test_paths)
    tasks = []
    for replication, seed in enumerate(seeds, start=1):
        if replication in completed:
            continue
        tasks.append(
            {
                "replication": replication,
                "seed": seed,
                "n_paths": n_paths,
                "n_steps": args.steps,
                "window_size": args.window,
                "p_switch": args.p_switch,
                "n_estimators": args.trees,
                "model_jobs": -1 if args.jobs == 1 else 1,
                "event_tolerance": args.event_tolerance,
                "evaluate_detector": not args.skip_detector,
                "signature_kind": args.signature_kind,
                "learners": tuple(args.learners),
            }
        )

    def checkpoint(row: Dict[str, float]) -> None:
        rows.append(row)
        current = pd.DataFrame(rows).sort_values("replication").reset_index(drop=True)
        save_result_tables(
            current,
            replications_path,
            summary_path,
            class_counts_path,
            diagnostics_path,
        )
        counts = "train={}/{}/{} validation={}/{}/{} test={}/{}/{}".format(
            *[
                int(row["{}_n{}".format(split_name, regime)])
                for split_name in ("train", "validation", "test")
                for regime in REGIME_IDS
            ]
        )
        detector_line = ""
        if "combined_test_boundary_f1" in row:
            detector_line = (
                "\n  boundary-F1(stat/sig/comb)={:.3f}/{:.3f}/{:.3f} "
                "delay={:.1f}/{:.1f}/{:.1f} false/1000={:.2f}/{:.2f}/{:.2f}".format(
                    *[
                        row[representation + "_test_" + metric]
                        for metric in (
                            "boundary_f1",
                            "mean_detection_delay",
                            "false_switches_per_1000",
                        )
                        for representation in REPRESENTATIONS
                    ]
                )
            )
        print(
            "Replication {}/{} seed={} {} ({:.1f}s)\n  {}{}".format(
                int(row["replication"]),
                args.replications,
                int(row["seed"]),
                progress_line(row, args.learners),
                row["elapsed_seconds"],
                counts,
                detector_line,
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
    summary = summarize_results(results)
    save_result_tables(
        results,
        replications_path,
        summary_path,
        class_counts_path,
        diagnostics_path,
    )

    print("\nPaired Experiment A summary")
    print(summary.to_string(index=False, float_format=lambda value: "{:.6f}".format(value)))
    print("\nSaved replications: {}".format(replications_path))
    print("Saved summary:      {}".format(summary_path))
    print("Saved class counts: {}".format(class_counts_path))
    print("Saved diagnostics:  {}".format(diagnostics_path))
    print("Total time: {:.1f}s".format(time.monotonic() - started))


if __name__ == "__main__":
    main()
