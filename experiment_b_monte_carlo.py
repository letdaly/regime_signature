"""Paired Monte Carlo study for Experiment B (matched marginals, new dynamics).

Each simulated path uses three CIR variance regimes with a common theta and a
common sigma^2/kappa ratio.  Their stationary one-time variance distribution is
therefore identical, while their mean-reversion speeds differ.  Statistics,
Signature, and Combined models use the same paths and Random Forest settings.

Detector smoothing, confidence, and confirmation settings are selected only on
validation paths.  Test paths are then evaluated once with state and event
metrics.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from experiment_a_monte_carlo import (
    DEFAULT_LEARNERS,
    DEFAULT_SIGNATURE_KIND,
    LEARNERS,
    LOGISTIC_C_GRID,
    REGIME_IDS,
    REPRESENTATIONS,
    add_learner_argument,
    add_path_count_arguments,
    add_signature_kind_argument,
    feature_indices,
    guard_output_directory,
    learner_contrasts,
    mean_t_interval,
    progress_line,
    replication_seeds,
    representation_contrasts,
    result_prefix,
    score_cell,
    split_class_counts,
)
from regime_detection import (
    DETECTOR_GRID,
    TIME_SINCE_CHANGE_BUCKETS,
    time_since_last_change,
)
from regime_pipeline import (
    CausalFeatureExtractor,
    build_path_level_splits,
    sample_matched_marginal_parameters,
)


CALIBRATION_BINS = 10


def classwise_probability_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    prefix: str,
    n_bins: int = CALIBRATION_BINS,
) -> Dict[str, float]:
    """Return classwise discrimination, frequency, and calibration diagnostics."""

    if probabilities.shape != (len(y_true), len(REGIME_IDS)):
        raise ValueError("probabilities must have one fixed-order column per regime.")
    if n_bins < 2:
        raise ValueError("n_bins must be at least two.")

    precisions = precision_score(
        y_true, y_pred, labels=REGIME_IDS, average=None, zero_division=0
    )
    recalls = recall_score(
        y_true, y_pred, labels=REGIME_IDS, average=None, zero_division=0
    )
    class_f1 = f1_score(
        y_true, y_pred, labels=REGIME_IDS, average=None, zero_division=0
    )
    diagnostics: Dict[str, float] = {}
    for regime in REGIME_IDS:
        binary_truth = (y_true == regime).astype(int)
        class_probability = probabilities[:, regime]
        diagnostics[f"{prefix}_precision_{regime}"] = float(precisions[regime])
        diagnostics[f"{prefix}_recall_{regime}"] = float(recalls[regime])
        diagnostics[f"{prefix}_f1_{regime}"] = float(class_f1[regime])
        diagnostics[f"{prefix}_prevalence_{regime}"] = float(np.mean(binary_truth))
        diagnostics[f"{prefix}_predicted_proportion_{regime}"] = float(
            np.mean(y_pred == regime)
        )
        diagnostics[f"{prefix}_prediction_excess_{regime}"] = float(
            np.mean(y_pred == regime) - np.mean(binary_truth)
        )
        diagnostics[f"{prefix}_brier_{regime}"] = float(
            np.mean((class_probability - binary_truth) ** 2)
        )
        if np.unique(binary_truth).size == 2:
            diagnostics[f"{prefix}_roc_auc_{regime}"] = float(
                roc_auc_score(binary_truth, class_probability)
            )
            diagnostics[f"{prefix}_pr_auc_{regime}"] = float(
                average_precision_score(binary_truth, class_probability)
            )
        else:
            diagnostics[f"{prefix}_roc_auc_{regime}"] = np.nan
            diagnostics[f"{prefix}_pr_auc_{regime}"] = np.nan

        bin_indices = np.minimum(
            (class_probability * n_bins).astype(int), n_bins - 1
        )
        for bin_index in range(n_bins):
            mask = bin_indices == bin_index
            diagnostics[
                f"{prefix}_calibration_{regime}_{bin_index}_count"
            ] = int(np.sum(mask))
            diagnostics[
                f"{prefix}_calibration_{regime}_{bin_index}_probability_sum"
            ] = float(np.sum(class_probability[mask]))
            diagnostics[
                f"{prefix}_calibration_{regime}_{bin_index}_positive_count"
            ] = int(np.sum(binary_truth[mask]))
    return diagnostics


CLASSWISE_CONTRAST_METRICS = (
    "recall",
    "precision",
    "f1",
    "predicted_proportion",
    "prediction_excess",
    "roc_auc",
    "pr_auc",
    "brier",
)


def evaluate_paired_replication(
    replication: int,
    seed: int,
    n_paths: Tuple[int, int, int],
    n_steps: int,
    window_size: int,
    p_switch: float,
    n_estimators: int,
    event_tolerance: int,
    model_jobs: int = -1,
    min_dwell: int = None,
    representations: Sequence[str] = REPRESENTATIONS,
    evaluate_detector: bool = True,
    signature_kind: str = DEFAULT_SIGNATURE_KIND,
    learners: Tuple[str, ...] = DEFAULT_LEARNERS,
) -> Dict[str, float]:
    """Evaluate every learner-representation cell on one matched-marginal path collection."""

    extractor = CausalFeatureExtractor(
        window_size=window_size,
        short_window=min(10, window_size),
        signature_level=3,
        representation="combined",
        signature_kind=signature_kind,
    )
    splits = build_path_level_splits(
        extractor=extractor,
        parameter_sampler=sample_matched_marginal_parameters,
        n_paths=n_paths,
        n_steps=n_steps,
        master_seed=seed,
        p_switch=p_switch,
        min_dwell=window_size if min_dwell is None else min_dwell,
        stationary_initial_variance=True,
        simulation_substeps=4,
    )
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

    unknown_representations = set(representations) - set(REPRESENTATIONS)
    if unknown_representations:
        raise ValueError(
            "Unknown representations: {}".format(sorted(unknown_representations))
        )

    for learner in learners:
        for representation in representations:
            row.update(
                score_cell(
                    splits,
                    learner,
                    representation,
                    selections[representation],
                    test_distances,
                    seed=seed,
                    n_estimators=n_estimators,
                    model_jobs=model_jobs,
                    event_tolerance=event_tolerance,
                    evaluate_detector=evaluate_detector,
                    extra_diagnostics=classwise_probability_diagnostics,
                )
            )
        if "statistics" not in representations:
            continue
        # Balanced-accuracy, time-since-change, and detector contrasts are the
        # shared ones; the classwise probability metrics are specific to this row.
        for candidate in ("combined", "signature"):
            if candidate not in representations:
                continue
            row.update(
                {
                    key: value
                    for key, value in representation_contrasts(
                        row, learner, evaluate_detector
                    ).items()
                    if key.startswith(result_prefix(learner, candidate) + "_minus_statistics")
                }
            )
            for metric in CLASSWISE_CONTRAST_METRICS:
                for regime in REGIME_IDS:
                    name = "{}_{}".format(metric, regime)
                    row[result_prefix(learner, candidate) + "_minus_statistics_" + name] = (
                        row[result_prefix(learner, candidate) + "_test_" + name]
                        - row[result_prefix(learner, "statistics") + "_test_" + name]
                    )
    if set(REPRESENTATIONS) <= set(representations):
        row.update(learner_contrasts(row, learners))
    return row


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    metrics = {}
    # Forest rows keep their historical labels ("BA_statistics"); the logistic
    # learner's rows carry its prefix ("BA_logistic_statistics").
    for learner in LEARNERS:
        cells = {
            representation: result_prefix(learner, representation)
            for representation in REPRESENTATIONS
        }
        for cell in cells.values():
            metrics["BA_" + cell] = cell + "_test_ba"
        for label, metric in (
            ("Detector_BA", "detector_balanced_accuracy"),
            ("Boundary_F1", "boundary_f1"),
            ("Detection_probability", "detection_probability"),
            ("False_switches_per_1000", "false_switches_per_1000"),
            ("Mean_delay", "mean_detection_delay"),
        ):
            for cell in cells.values():
                metrics[label + "_" + cell] = cell + "_test_" + metric
        metrics["Delta_BA_" + cells["combined"] + "_minus_statistics"] = (
            cells["combined"] + "_minus_statistics_ba"
        )
        for name, _, _ in TIME_SINCE_CHANGE_BUCKETS:
            for cell in cells.values():
                metrics["BA_since_{}_{}".format(name, cell)] = cell + "_test_ba_since_" + name
            metrics["Delta_BA_since_{}_{}_minus_statistics".format(name, cells["combined"])] = (
                cells["combined"] + "_minus_statistics_ba_since_" + name
            )
        for label, metric in (
            ("detector_BA", "detector_balanced_accuracy"),
            ("boundary_F1", "boundary_f1"),
            ("detection_probability", "detection_probability"),
            ("false_switches", "false_switches_per_1000"),
            ("mean_delay", "mean_detection_delay"),
        ):
            metrics["Delta_{}_{}_minus_statistics".format(label, cells["combined"])] = (
                cells["combined"] + "_minus_statistics_" + metric
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


def cells_present(row: pd.Series) -> Sequence[Tuple[str, str]]:
    """Learner-representation cells that this replication row actually scored."""

    return [
        (learner, representation)
        for learner in LEARNERS
        for representation in REPRESENTATIONS
        if result_prefix(learner, representation) + "_test_ba" in row
    ]


def save_outputs(results: pd.DataFrame, output_dir: Path) -> None:
    results.to_csv(output_dir / "replications.csv", index=False)
    if len(results) >= 2:
        summarize_results(results).to_csv(output_dir / "summary.csv", index=False)

    count_rows = []
    diagnostic_rows = []
    classwise_rows = []
    calibration_rows = []
    for _, row in results.iterrows():
        identity = {"replication": int(row["replication"]), "seed": int(row["seed"])}
        for split_name in ("train", "validation", "test"):
            count_rows.append(
                {
                    **identity,
                    "split": split_name,
                    **{
                        "n{}".format(regime): int(row["{}_n{}".format(split_name, regime)])
                        for regime in REGIME_IDS
                    },
                }
            )
        for learner, representation in cells_present(row):
            cell = result_prefix(learner, representation)
            prefix = cell + "_test_"
            # ``learner`` is carried on every tidy row so the classwise and
            # calibration tables can be grouped by learner as well as by
            # representation.
            identity["learner"] = learner
            diagnostic = {
                    **identity,
                    "representation": representation,
                    "balanced_accuracy": row[prefix + "ba"],
                    "macro_f1": row[prefix + "macro_f1"],
                    "detector_balanced_accuracy": row[prefix + "detector_balanced_accuracy"],
                    "fraction_time_wrong": row[prefix + "fraction_time_wrong"],
                    "brier_score": row[prefix + "brier_score"],
                    "detection_probability": row[prefix + "detection_probability"],
                    "mean_detection_delay": row[prefix + "mean_detection_delay"],
                    "false_switches_per_1000": row[prefix + "false_switches_per_1000"],
                    "boundary_f1": row[prefix + "boundary_f1"],
                    "alpha": row[cell + "_detector_alpha"],
                    "threshold": row[cell + "_detector_threshold"],
                    "confirmations": int(row[cell + "_detector_confirmations"]),
            }
            if learner == "logistic":
                diagnostic["selected_c"] = row[cell + "_selected_c"]
            for name, _, _ in TIME_SINCE_CHANGE_BUCKETS:
                if prefix + "ba_since_" + name in row:
                    diagnostic["ba_since_" + name] = row[prefix + "ba_since_" + name]
                    diagnostic["n_since_" + name] = int(row[prefix + "n_since_" + name])
            for regime in REGIME_IDS:
                for metric in (
                    "recall",
                    "precision",
                    "f1",
                    "prevalence",
                    "predicted_proportion",
                    "prediction_excess",
                    "roc_auc",
                    "pr_auc",
                    "brier",
                ):
                    diagnostic[f"{metric}_{regime}"] = row[
                        f"{prefix}{metric}_{regime}"
                    ]
                classwise_rows.append(
                    {
                        **identity,
                        "representation": representation,
                        "regime": regime,
                        **{
                            metric: row[f"{prefix}{metric}_{regime}"]
                            for metric in (
                                "recall",
                                "precision",
                                "f1",
                                "prevalence",
                                "predicted_proportion",
                                "prediction_excess",
                                "roc_auc",
                                "pr_auc",
                                "brier",
                            )
                        },
                    }
                )
                for bin_index in range(CALIBRATION_BINS):
                    calibration_rows.append(
                        {
                            **identity,
                            "representation": representation,
                            "regime": regime,
                            "bin": bin_index,
                            "bin_low": bin_index / CALIBRATION_BINS,
                            "bin_high": (bin_index + 1) / CALIBRATION_BINS,
                            "count": int(
                                row[f"{prefix}calibration_{regime}_{bin_index}_count"]
                            ),
                            "probability_sum": row[
                                f"{prefix}calibration_{regime}_{bin_index}_probability_sum"
                            ],
                            "positive_count": int(
                                row[
                                    f"{prefix}calibration_{regime}_{bin_index}_positive_count"
                                ]
                            ),
                        }
                    )
            diagnostic_rows.append(diagnostic)
    pd.DataFrame(count_rows).to_csv(output_dir / "class_counts.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(output_dir / "test_diagnostics.csv", index=False)
    classwise = pd.DataFrame(classwise_rows)
    classwise.to_csv(output_dir / "classwise_metrics.csv", index=False)

    summary_rows = []
    for (learner, representation, regime), group in classwise.groupby(
        ["learner", "representation", "regime"], sort=False
    ):
        for metric in (
            "recall",
            "precision",
            "f1",
            "prevalence",
            "predicted_proportion",
            "prediction_excess",
            "roc_auc",
            "pr_auc",
            "brier",
        ):
            values = group[metric].dropna()
            if len(values) >= 2:
                summary_rows.append(
                    {
                        "learner": learner,
                        "representation": representation,
                        "regime": int(regime),
                        "metric": metric,
                        **mean_t_interval(values),
                    }
                )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "classwise_summary.csv", index=False
    )

    contrast_rows = []
    for (learner, regime), regime_rows in classwise.groupby(["learner", "regime"], sort=False):
        for candidate in ("signature", "combined"):
            for metric in (
                "recall",
                "precision",
                "f1",
                "predicted_proportion",
                "prediction_excess",
                "roc_auc",
                "pr_auc",
                "brier",
            ):
                wide = regime_rows.pivot(
                    index="replication", columns="representation", values=metric
                )
                if candidate not in wide or "statistics" not in wide:
                    continue
                differences = (wide[candidate] - wide["statistics"]).dropna()
                if len(differences) >= 2:
                    contrast_rows.append(
                        {
                            "learner": learner,
                            "candidate": candidate,
                            "baseline": "statistics",
                            "regime": regime,
                            "metric": metric,
                            **mean_t_interval(differences),
                            "candidate_win_rate": float(np.mean(differences > 0)),
                        }
                    )
    pd.DataFrame(contrast_rows).to_csv(
        output_dir / "classwise_contrasts.csv", index=False
    )

    calibration = pd.DataFrame(calibration_rows)
    aggregated = (
        calibration.groupby(
            ["learner", "representation", "regime", "bin", "bin_low", "bin_high"],
            as_index=False,
        )[["count", "probability_sum", "positive_count"]]
        .sum()
    )
    aggregated["mean_predicted_probability"] = np.where(
        aggregated["count"] > 0,
        aggregated["probability_sum"] / aggregated["count"],
        np.nan,
    )
    aggregated["observed_frequency"] = np.where(
        aggregated["count"] > 0,
        aggregated["positive_count"] / aggregated["count"],
        np.nan,
    )
    aggregated.to_csv(output_dir / "calibration_bins.csv", index=False)


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
    parser.add_argument("--event-tolerance", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=20260828)
    parser.add_argument("--jobs", type=int, default=1)
    # Main-table grid directory; the frozen 3-validation-path study is
    # results/experiment_b_diagnostics and is never written to.
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/main_grid/heston_matched_marginal")
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _evaluate_task(task: Dict[str, object]) -> Dict[str, float]:
    started = time.monotonic()
    row = evaluate_paired_replication(**task)
    row["elapsed_seconds"] = float(time.monotonic() - started)
    return row


def main() -> None:
    args = parse_args()
    if args.replications < 2:
        raise ValueError("Experiment B requires at least two replications.")
    if min(args.train_paths, args.validation_paths, args.test_paths) < 1:
        raise ValueError("Every split needs at least one path.")
    if args.jobs < 1 or args.event_tolerance < 1:
        raise ValueError("jobs and event-tolerance must be positive.")

    output_dir = args.output_dir.resolve()
    guard_output_directory(output_dir, args.resume)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "scenario": "matched_stationary_cir_marginal_different_kappa",
        "replications": args.replications,
        "n_paths": [args.train_paths, args.validation_paths, args.test_paths],
        "n_steps": args.steps,
        "window_size": args.window,
        "p_switch": args.p_switch,
        "n_estimators": args.trees,
        "event_tolerance": args.event_tolerance,
        "base_seed": args.base_seed,
        "signature_kind": args.signature_kind,
        "signature_level": 3,
        "signature_path": CausalFeatureExtractor.PATH_CONVENTION,
        "learners": list(args.learners),
        "logistic_c_grid": list(LOGISTIC_C_GRID),
        "simulation_substeps": 4,
        "classwise_probability_diagnostics": True,
        "calibration_bins": CALIBRATION_BINS,
        "detector_grid": [list(values) for values in DETECTOR_GRID],
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
        completed = set(existing["replication"].astype(int))
        rows = existing.to_dict("records")
    else:
        completed = set()
        rows = []

    seeds = replication_seeds(args.base_seed, args.replications)
    tasks = [
        {
            "replication": replication,
            "seed": seed,
            "n_paths": (args.train_paths, args.validation_paths, args.test_paths),
            "n_steps": args.steps,
            "window_size": args.window,
            "p_switch": args.p_switch,
            "n_estimators": args.trees,
            "event_tolerance": args.event_tolerance,
            "model_jobs": -1 if args.jobs == 1 else 1,
            "signature_kind": args.signature_kind,
            "learners": tuple(args.learners),
        }
        for replication, seed in enumerate(seeds, start=1)
        if replication not in completed
    ]

    def checkpoint(row: Dict[str, float]) -> None:
        rows.append(row)
        current = pd.DataFrame(rows).sort_values("replication").reset_index(drop=True)
        save_outputs(current, output_dir)
        first = args.learners[0]
        print(
            "Replication {}/{} seed={} {} boundary-F1({})={:.3f}/{:.3f}/{:.3f} ({:.1f}s)".format(
                int(row["replication"]),
                args.replications,
                int(row["seed"]),
                progress_line(row, args.learners),
                first,
                *[
                    row[result_prefix(first, representation) + "_test_boundary_f1"]
                    for representation in REPRESENTATIONS
                ],
                row["elapsed_seconds"],
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
    save_outputs(results, output_dir)
    summary = summarize_results(results)
    print("\nPaired Experiment B summary")
    print(summary.to_string(index=False, float_format=lambda value: "{:.6f}".format(value)))
    print("\nSaved results to {}".format(output_dir))
    print("Total time: {:.1f}s".format(time.monotonic() - started))


if __name__ == "__main__":
    main()
