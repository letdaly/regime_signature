"""Signature truncation-order ablation with a noise-padded Statistics control.

For each replication of a main-grid Heston design, the path collection and one
feature matrix (11 Statistics columns plus the level-4 log-signature, 32
coordinates) are built once.  Because iisignature orders log-signature
coordinates by level, the level-2 and level-3 log-signatures are the first
6 and 14 of those 32 columns, so every truncation level is a column subset of
the same matrix.  Per level L with d_L coordinates the representations are

- ``signature_L``: the d_L log-signature coordinates alone;
- ``combined_L``: Statistics plus those coordinates;
- ``noise_L``: Statistics plus d_L i.i.d. standard-normal columns, drawn
  independently for every window from the replication seed.

``noise_L`` is the dimension control: a learner that gains from
``combined_L`` only because it received more columns should gain as much from
``noise_L``.  Both main-grid learners are fitted to every representation with
the same code as the main rows (``experiment_a_monte_carlo.fit_learner``);
classification is scored pointwise and by time since the last true change.
The causal detector is not run here.

Contrasts, per learner and level: ``combined_L - statistics``,
``combined_L - noise_L``, ``signature_L - statistics``, and
``combined_L - combined_{L-1}`` between successive levels.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from experiment_a_monte_carlo import (
    DEFAULT_LEARNERS,
    LEARNERS,
    LOGISTIC_C_GRID,
    REGIME_IDS,
    add_learner_argument,
    add_path_count_arguments,
    classification_diagnostics,
    fit_learner,
    guard_output_directory,
    mean_t_interval,
    replication_seeds,
    result_prefix,
    split_class_counts,
)
from regime_detection import (
    TIME_SINCE_CHANGE_BUCKETS,
    ordered_probabilities,
    stratified_classification,
    time_since_last_change,
)
from regime_pipeline import (
    CausalFeatureExtractor,
    PathLevelSplits,
    WindowDataset,
    build_path_level_splits,
    sample_matched_marginal_parameters,
    sample_scale_only_parameters,
)


DESIGNS: Dict[str, Dict[str, object]] = {
    "scale_only": {
        "sampler": sample_scale_only_parameters,
        "substeps": 1,
        "base_seed": 20260827,
        "heston_results": "results/main_grid/heston_scale_only",
    },
    "matched_marginal": {
        "sampler": sample_matched_marginal_parameters,
        "substeps": 4,
        "base_seed": 20260828,
        "heston_results": "results/main_grid/heston_matched_marginal",
    },
}
DEFAULT_LEVELS = (2, 3, 4)
LOGSIGNATURE_LENGTHS = {1: 3, 2: 6, 3: 14, 4: 32}


def representation_names(levels: Sequence[int]) -> Tuple[str, ...]:
    names = ["statistics"]
    for level in levels:
        names.extend(
            ("signature_{}".format(level), "combined_{}".format(level), "noise_{}".format(level))
        )
    return tuple(names)


def pad_with_noise(dataset: WindowDataset, n_noise: int, rng: np.random.Generator) -> WindowDataset:
    """Append ``n_noise`` independent standard-normal columns to a window dataset."""

    noise = rng.standard_normal((len(dataset), n_noise))
    names = dataset.feature_names + tuple("noise_{}".format(i) for i in range(n_noise))
    return replace(dataset, X=np.hstack((dataset.X, noise)), feature_names=names)


def column_selections(
    feature_names: Iterable[str], levels: Sequence[int]
) -> Dict[str, np.ndarray]:
    """Column subsets of the padded matrix for every representation."""

    names = tuple(feature_names)
    statistics = np.array(
        [i for i, name in enumerate(names) if not name.startswith(("logsignature_", "noise_"))],
        dtype=int,
    )
    signature = np.array(
        [i for i, name in enumerate(names) if name.startswith("logsignature_")], dtype=int
    )
    noise = np.array([i for i, name in enumerate(names) if name.startswith("noise_")], dtype=int)
    max_length = LOGSIGNATURE_LENGTHS[max(levels)]
    if len(signature) != max_length or len(noise) != max_length:
        raise ValueError(
            "Expected {} log-signature and noise columns, found {} and {}.".format(
                max_length, len(signature), len(noise)
            )
        )
    selections = {"statistics": statistics}
    for level in levels:
        d = LOGSIGNATURE_LENGTHS[level]
        selections["signature_{}".format(level)] = signature[:d]
        selections["combined_{}".format(level)] = np.concatenate((statistics, signature[:d]))
        selections["noise_{}".format(level)] = np.concatenate((statistics, noise[:d]))
    return selections


def build_ablation_splits(
    design: str,
    seed: int,
    n_paths: Tuple[int, int, int],
    n_steps: int,
    window_size: int,
    p_switch: float,
    levels: Sequence[int],
) -> PathLevelSplits:
    """Simulate one replication and build the level-max(levels) padded matrix."""

    settings = DESIGNS[design]
    extractor = CausalFeatureExtractor(
        window_size=window_size,
        short_window=min(10, window_size),
        signature_level=max(levels),
        representation="combined",
        signature_kind="log",
    )
    splits = build_path_level_splits(
        extractor=extractor,
        parameter_sampler=settings["sampler"],
        n_paths=n_paths,
        n_steps=n_steps,
        master_seed=seed,
        p_switch=p_switch,
        min_dwell=window_size,
        stationary_initial_variance=True,
        simulation_substeps=int(settings["substeps"]),
    )
    # One noise stream per replication, consumed in a fixed split order so the
    # control is reproducible from the seed alone.
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xAB1A]))
    n_noise = LOGSIGNATURE_LENGTHS[max(levels)]
    return PathLevelSplits(
        train=pad_with_noise(splits.train, n_noise, rng),
        validation=pad_with_noise(splits.validation, n_noise, rng),
        test=pad_with_noise(splits.test, n_noise, rng),
        paths=splits.paths,
    )


def evaluate_paired_replication(
    replication: int,
    seed: int,
    design: str,
    n_paths: Tuple[int, int, int],
    n_steps: int,
    window_size: int,
    p_switch: float,
    n_estimators: int,
    levels: Sequence[int] = DEFAULT_LEVELS,
    learners: Tuple[str, ...] = DEFAULT_LEARNERS,
    model_jobs: int = -1,
) -> Dict[str, float]:
    """Score every learner on every truncation-level representation of one replication."""

    levels = tuple(sorted(int(level) for level in levels))
    splits = build_ablation_splits(
        design, seed, n_paths, n_steps, window_size, p_switch, levels
    )
    splits.assert_disjoint()
    selections = column_selections(splits.train.feature_names, levels)
    test_distances = time_since_last_change(splits.test, splits.paths["test"])

    row: Dict[str, float] = {
        "replication": int(replication),
        "seed": int(seed),
        "n_train_windows": int(len(splits.train)),
        "n_validation_windows": int(len(splits.validation)),
        "n_test_windows": int(len(splits.test)),
    }
    row.update(split_class_counts(splits))

    for learner in learners:
        for representation, columns in selections.items():
            prefix = result_prefix(learner, representation)
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
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                validation_prediction = np.argmax(
                    ordered_probabilities(model, splits.validation.X[:, columns]), axis=1
                )
                test_prediction = np.argmax(
                    ordered_probabilities(model, splits.test.X[:, columns]), axis=1
                )
            row[prefix + "_n_features"] = int(len(columns))
            row[prefix + "_validation_ba"] = float(
                balanced_accuracy_score(splits.validation.y, validation_prediction)
            )
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
        row.update(level_contrasts(row, learner, levels))
    return row


def contrast_pairs(levels: Sequence[int]) -> Tuple[Tuple[str, str], ...]:
    """(candidate, baseline) representation pairs reported for every learner."""

    pairs = []
    for index, level in enumerate(levels):
        pairs.append(("combined_{}".format(level), "statistics"))
        pairs.append(("combined_{}".format(level), "noise_{}".format(level)))
        pairs.append(("signature_{}".format(level), "statistics"))
        pairs.append(("noise_{}".format(level), "statistics"))
        if index > 0:
            pairs.append(("combined_{}".format(level), "combined_{}".format(levels[index - 1])))
    return tuple(pairs)


def level_contrasts(
    row: Dict[str, float], learner: str, levels: Sequence[int]
) -> Dict[str, float]:
    contrasts: Dict[str, float] = {}
    for candidate, baseline in contrast_pairs(levels):
        key = result_prefix(learner, candidate) + "_minus_" + baseline
        contrasts[key + "_ba"] = (
            row[result_prefix(learner, candidate) + "_test_ba"]
            - row[result_prefix(learner, baseline) + "_test_ba"]
        )
        for name, _, _ in TIME_SINCE_CHANGE_BUCKETS:
            contrasts[key + "_ba_since_" + name] = (
                row[result_prefix(learner, candidate) + "_test_ba_since_" + name]
                - row[result_prefix(learner, baseline) + "_test_ba_since_" + name]
            )
    return contrasts


def summarize_results(results: pd.DataFrame, levels: Sequence[int]) -> pd.DataFrame:
    metrics = {}
    for learner in LEARNERS:
        for representation in representation_names(levels):
            cell = result_prefix(learner, representation)
            metrics["BA_" + cell] = cell + "_test_ba"
        for candidate, baseline in contrast_pairs(levels):
            key = result_prefix(learner, candidate) + "_minus_" + baseline
            metrics["Delta_" + key] = key + "_ba"
            for name, _, _ in TIME_SINCE_CHANGE_BUCKETS:
                metrics["Delta_BA_since_{}_{}".format(name, key)] = key + "_ba_since_" + name
    rows = []
    for label, column in metrics.items():
        if column not in results.columns:
            continue
        values = results[column].dropna()
        if len(values) >= 2:
            summary = {"metric": label, **mean_t_interval(values)}
            if label.startswith("Delta_") and "since" not in label:
                summary["positive_share"] = float(np.mean(values > 0))
            rows.append(summary)
    return pd.DataFrame(rows)


def diagnostic_table(results: pd.DataFrame, levels: Sequence[int]) -> pd.DataFrame:
    rows = []
    for _, row in results.iterrows():
        for learner in LEARNERS:
            for representation in representation_names(levels):
                cell = result_prefix(learner, representation)
                if cell + "_test_ba" not in row:
                    continue
                record = {
                    "replication": int(row["replication"]),
                    "seed": int(row["seed"]),
                    "learner": learner,
                    "representation": representation,
                    "n_features": int(row[cell + "_n_features"]),
                    "validation_ba": row[cell + "_validation_ba"],
                    "balanced_accuracy": row[cell + "_test_ba"],
                    "macro_f1": row[cell + "_test_macro_f1"],
                }
                if learner == "logistic":
                    record["selected_c"] = row[cell + "_selected_c"]
                for regime in REGIME_IDS:
                    record["recall_{}".format(regime)] = row[cell + "_test_recall_{}".format(regime)]
                for name, _, _ in TIME_SINCE_CHANGE_BUCKETS:
                    record["ba_since_" + name] = row[cell + "_test_ba_since_" + name]
                rows.append(record)
    return pd.DataFrame(rows)


def save_outputs(results: pd.DataFrame, output_dir: Path, levels: Sequence[int]) -> None:
    results.to_csv(output_dir / "replications.csv", index=False)
    diagnostic_table(results, levels).to_csv(output_dir / "test_diagnostics.csv", index=False)
    if len(results) >= 2:
        summarize_results(results, levels).to_csv(output_dir / "summary.csv", index=False)


def _evaluate_task(task: Dict[str, object]) -> Dict[str, float]:
    started = time.monotonic()
    row = evaluate_paired_replication(**task)
    row["elapsed_seconds"] = float(time.monotonic() - started)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--design", required=True, choices=sorted(DESIGNS))
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=list(DEFAULT_LEVELS),
        help="Log-signature truncation levels to compare (each in 1-4).",
    )
    parser.add_argument("--replications", type=int, default=50)
    add_path_count_arguments(parser)
    add_learner_argument(parser)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--p-switch", type=float, default=0.002)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help="Defaults to the design's main-grid base seed (paired parameter draws).",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to results/main_grid/truncation_<design>.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    levels = tuple(sorted(set(int(level) for level in args.levels)))
    if any(level not in LOGSIGNATURE_LENGTHS for level in levels):
        raise ValueError("levels must lie in 1-4.")
    if args.replications < 2:
        raise ValueError("At least two replications are required.")
    if min(args.train_paths, args.validation_paths, args.test_paths) < 1:
        raise ValueError("Every split needs at least one path.")
    if args.jobs < 1:
        raise ValueError("jobs must be positive.")

    design = DESIGNS[args.design]
    base_seed = int(design["base_seed"] if args.base_seed is None else args.base_seed)
    output_dir = (
        Path("results/main_grid/truncation_{}".format(args.design))
        if args.output_dir is None
        else args.output_dir
    ).resolve()
    guard_output_directory(output_dir, args.resume)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "experiment": "signature_truncation_ablation",
        "design": args.design,
        "levels": list(levels),
        "logsignature_lengths": {str(level): LOGSIGNATURE_LENGTHS[level] for level in levels},
        "noise_control": "Statistics plus d_L i.i.d. standard-normal columns per window",
        "replications": args.replications,
        "n_paths": [args.train_paths, args.validation_paths, args.test_paths],
        "n_steps": args.steps,
        "window_size": args.window,
        "p_switch": args.p_switch,
        "n_estimators": args.trees,
        "simulation_substeps": int(design["substeps"]),
        "base_seed": base_seed,
        "paired_parameter_draws_with": design["heston_results"]
        if base_seed == design["base_seed"]
        else None,
        "signature_kind": "log",
        "signature_path": CausalFeatureExtractor.PATH_CONVENTION,
        "learners": list(args.learners),
        "logistic_c_grid": list(LOGISTIC_C_GRID),
        "evaluate_detector": False,
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

    seeds = replication_seeds(base_seed, args.replications)
    tasks = [
        {
            "replication": replication,
            "seed": seed,
            "design": args.design,
            "n_paths": (args.train_paths, args.validation_paths, args.test_paths),
            "n_steps": args.steps,
            "window_size": args.window,
            "p_switch": args.p_switch,
            "n_estimators": args.trees,
            "levels": levels,
            "learners": tuple(args.learners),
            "model_jobs": -1 if args.jobs == 1 else 1,
        }
        for replication, seed in enumerate(seeds, start=1)
        if replication not in completed
    ]

    def checkpoint(row: Dict[str, float]) -> None:
        rows.append(row)
        current = pd.DataFrame(rows).sort_values("replication").reset_index(drop=True)
        save_outputs(current, output_dir, levels)
        parts = []
        for learner in args.learners:
            cells = " ".join(
                "L{}={:+.4f}/{:+.4f}".format(
                    level,
                    row[result_prefix(learner, "combined_{}".format(level)) + "_minus_statistics_ba"],
                    row[result_prefix(learner, "combined_{}".format(level)) + "_minus_noise_{}_ba".format(level)],
                )
                for level in levels
            )
            parts.append("{} stat={:.4f} comb-stat/comb-noise {}".format(
                "RF" if learner == "random_forest" else "logit",
                row[result_prefix(learner, "statistics") + "_test_ba"],
                cells,
            ))
        print(
            "Replication {}/{} seed={} {} ({:.1f}s)".format(
                int(row["replication"]), args.replications, int(row["seed"]),
                " | ".join(parts), row["elapsed_seconds"],
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
    save_outputs(results, output_dir, levels)
    print("\nTruncation ablation summary ({})".format(args.design))
    print(
        summarize_results(results, levels).to_string(
            index=False, float_format=lambda value: "{:.6f}".format(value)
        )
    )
    print("\nSaved results to {}".format(output_dir))
    print("Total time: {:.1f}s".format(time.monotonic() - started))


if __name__ == "__main__":
    main()
