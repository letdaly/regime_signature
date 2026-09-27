"""Leakage-resistant mechanism benchmark for Experiments A and B.

This runner implements the first synthetic stages of the Beyond Marginal
Regimes plan.  It compares a prespecified hierarchy of conventional summaries
with raw path signatures and a temporal-order placebo.  Every representation
uses identical simulated paths.  Scaling and model selection are fit on
training/validation paths only; test paths are evaluated once.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler

from experiment_a_monte_carlo import mean_t_interval, replication_seeds
from regime_pipeline import (
    CausalFeatureExtractor,
    PathLevelSplits,
    WindowDataset,
    build_path_level_splits,
    combine_window_datasets,
    sample_matched_marginal_parameters,
    sample_scale_only_parameters,
)


SCENARIOS = {
    "a_scale": sample_scale_only_parameters,
    "b_dynamics": sample_matched_marginal_parameters,
}
MODEL_FAMILIES = ("logistic", "random_forest")
ORIGINAL_REPRESENTATIONS = ("b1", "b2", "b3", "signature", "b3_signature")


def clip_scaled_features(values: np.ndarray) -> np.ndarray:
    """Bound training-scaled tails to keep linear optimization well conditioned."""

    return np.clip(values, -20.0, 20.0)


def preprocessing_steps():
    return [
        ("scale", RobustScaler(quantile_range=(10.0, 90.0), unit_variance=True)),
        ("clip", FunctionTransformer(clip_scaled_features)),
    ]


def feature_selections(feature_names: Iterable[str]) -> Dict[str, np.ndarray]:
    names = tuple(feature_names)
    prefixes = {
        "b1": ("b1_",),
        "b2": ("b1_", "b2_"),
        "b3": ("b1_", "b2_", "b3_"),
        "signature": ("signature_", "logsignature_"),
        "b3_signature": ("b1_", "b2_", "b3_", "signature_", "logsignature_"),
    }
    selections = {
        label: np.asarray(
            [i for i, name in enumerate(names) if name.startswith(allowed)], dtype=int
        )
        for label, allowed in prefixes.items()
    }
    if any(len(columns) == 0 for columns in selections.values()):
        raise ValueError("Rich combined features did not produce every benchmark block.")
    return selections


def ordered_probabilities(model: Pipeline, X: np.ndarray) -> np.ndarray:
    probabilities = np.zeros((len(X), 3), dtype=float)
    # NumPy 2.0 linked against Accelerate can emit false floating-point
    # matmul warnings for finite float64 operands. Validate the result instead.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        fitted = model.predict_proba(X)
    if not np.isfinite(fitted).all():
        raise FloatingPointError("Model returned non-finite probabilities.")
    for column, regime in enumerate(model.classes_.astype(int)):
        probabilities[:, regime] = fitted[:, column]
    return probabilities


def model_grid(family: str, seed: int, trees: int, jobs: int):
    if family == "logistic":
        for c_value in (0.01, 0.1, 1.0):
            yield "C={}".format(c_value), Pipeline(
                preprocessing_steps() + [
                    (
                        "model",
                        LogisticRegression(
                            C=c_value,
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=seed,
                            solver="liblinear",
                        ),
                    )
                ]
            )
    elif family == "random_forest":
        for depth in (4, 6, 10):
            yield "max_depth={}".format(depth), Pipeline(
                preprocessing_steps() + [
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=trees,
                            max_depth=depth,
                            class_weight="balanced",
                            random_state=seed,
                            n_jobs=jobs,
                        ),
                    )
                ]
            )
    else:
        raise ValueError("Unknown model family: {}".format(family))


def evaluate_once(
    train: WindowDataset,
    validation: WindowDataset,
    test: WindowDataset,
    columns: np.ndarray,
    family: str,
    seed: int,
    trees: int,
    jobs: int,
) -> Dict[str, float]:
    """Tune on validation only, then score the untouched test partition once."""

    best = None
    for setting, candidate in model_grid(family, seed, trees, jobs):
        candidate.fit(train.X[:, columns], train.y)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            validation_prediction = candidate.predict(validation.X[:, columns])
        score = float(balanced_accuracy_score(validation.y, validation_prediction))
        if best is None or score > best[0]:
            best = (score, setting, candidate)
    validation_ba, setting, model = best
    probabilities = ordered_probabilities(model, test.X[:, columns])
    prediction = np.argmax(probabilities, axis=1)
    targets = np.eye(3)[test.y]
    try:
        auc = float(
            roc_auc_score(test.y, probabilities, labels=(0, 1, 2), multi_class="ovr")
        )
    except ValueError:
        auc = np.nan
    return {
        "selected_setting": setting,
        "validation_ba": validation_ba,
        "test_ba": float(balanced_accuracy_score(test.y, prediction)),
        "test_macro_f1": float(
            f1_score(test.y, prediction, labels=(0, 1, 2), average="macro", zero_division=0)
        ),
        "test_roc_auc_ovr": auc,
        "test_log_loss": float(log_loss(test.y, probabilities, labels=(0, 1, 2))),
        "test_brier": float(np.mean(np.sum((probabilities - targets) ** 2, axis=1))),
    }


def extract_from_existing_paths(
    splits: PathLevelSplits, extractor: CausalFeatureExtractor
) -> PathLevelSplits:
    data = {
        split_name: combine_window_datasets(
            extractor.extract_path(path) for path in splits.paths[split_name]
        )
        for split_name in ("train", "validation", "test")
    }
    result = PathLevelSplits(
        train=data["train"],
        validation=data["validation"],
        test=data["test"],
        paths=splits.paths,
    )
    result.assert_disjoint()
    return result


def evaluate_replication(
    scenario: str,
    replication: int,
    seed: int,
    n_paths: Tuple[int, int, int],
    n_steps: int,
    window_size: int,
    p_switch: float,
    trees: int,
    model_families: Sequence[str],
    shuffle_repeats: int,
    block_size: int,
    jobs: int,
) -> List[Dict[str, object]]:
    original_extractor = CausalFeatureExtractor(
        window_size=window_size,
        short_window=min(10, window_size),
        signature_level=3,
        representation="combined",
        signature_kind="raw",
        statistics_level="rich",
    )
    splits = build_path_level_splits(
        extractor=original_extractor,
        parameter_sampler=SCENARIOS[scenario],
        n_paths=n_paths,
        n_steps=n_steps,
        master_seed=seed,
        p_switch=p_switch,
        min_dwell=window_size,
        stationary_initial_variance=True,
        simulation_substeps=1 if scenario == "a_scale" else 4,
    )
    selections = feature_selections(splits.train.feature_names)
    rows: List[Dict[str, object]] = []

    def add_result(
        representation: str,
        data: PathLevelSplits,
        columns: np.ndarray,
        placebo_repeat: int = 0,
    ) -> None:
        for family in model_families:
            metrics = evaluate_once(
                data.train,
                data.validation,
                data.test,
                columns,
                family,
                seed,
                trees,
                jobs,
            )
            rows.append(
                {
                    "scenario": scenario,
                    "replication": replication,
                    "seed": seed,
                    "model_family": family,
                    "representation": representation,
                    "placebo_repeat": placebo_repeat,
                    "n_features": len(columns),
                    "n_train_windows": len(data.train),
                    "n_validation_windows": len(data.validation),
                    "n_test_windows": len(data.test),
                    **metrics,
                }
            )

    for representation in ORIGINAL_REPRESENTATIONS:
        add_result(representation, splits, selections[representation])

    for repeat in range(1, shuffle_repeats + 1):
        for increment_order, label in (
            ("shuffle", "shuffled_signature"),
            ("block_shuffle", "block_shuffled_signature"),
        ):
            shuffled_extractor = CausalFeatureExtractor(
                window_size=window_size,
                short_window=min(10, window_size),
                signature_level=3,
                representation="signature",
                signature_kind="raw",
                statistics_level="rich",
                increment_order=increment_order,
                order_seed=seed + repeat,
                block_size=block_size,
            )
            shuffled = extract_from_existing_paths(splits, shuffled_extractor)
            columns = np.arange(len(shuffled.train.feature_names), dtype=int)
            add_result(label, shuffled, columns, placebo_repeat=repeat)
    return rows


def summarize(results: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    for keys, group in results.groupby(
        ["scenario", "model_family", "representation"], sort=True
    ):
        scenario, family, representation = keys
        if representation in ("shuffled_signature", "block_shuffled_signature"):
            values = group.groupby("replication")["test_ba"].mean()
        else:
            values = group["test_ba"]
        if len(values) < 2:
            continue
        summary_rows.append(
            {
                "scenario": scenario,
                "model_family": family,
                "representation": representation,
                **mean_t_interval(values),
            }
        )

    contrast_rows = []
    averaged = (
        results.groupby(
            ["scenario", "replication", "model_family", "representation"],
            as_index=False,
        )["test_ba"]
        .mean()
    )
    for (scenario, family), group in averaged.groupby(["scenario", "model_family"]):
        wide = group.pivot(index="replication", columns="representation", values="test_ba")
        for candidate, baseline, label in (
            ("b3_signature", "b3", "incremental_signature_over_b3"),
            ("signature", "shuffled_signature", "temporal_order_effect"),
            ("signature", "block_shuffled_signature", "long_range_order_effect"),
        ):
            differences = (wide[candidate] - wide[baseline]).dropna()
            if len(differences) < 2:
                continue
            contrast_rows.append(
                {
                    "scenario": scenario,
                    "model_family": family,
                    "contrast": label,
                    "candidate": candidate,
                    "baseline": baseline,
                    **mean_t_interval(differences),
                    "candidate_win_rate": float(np.mean(differences > 0)),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(contrast_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", choices=tuple(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--model-families", nargs="+", choices=MODEL_FAMILIES, default=list(MODEL_FAMILIES))
    parser.add_argument("--replications", type=int, default=20)
    parser.add_argument("--train-paths", type=int, default=10)
    parser.add_argument("--validation-paths", type=int, default=5)
    parser.add_argument("--test-paths", type=int, default=5)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--p-switch", type=float, default=0.002)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--shuffle-repeats", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=20260915)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--output-dir", type=Path, default=Path("results/beyond_marginal_benchmark"))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replications < 2 or args.shuffle_repeats < 2:
        raise ValueError("Use at least two replications and two shuffle repeats.")
    if args.block_size < 2 or args.block_size >= args.window:
        raise ValueError("block-size must be at least 2 and smaller than window.")
    if args.window < 10 or args.window > args.steps:
        raise ValueError("window must be between 10 and steps.")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "scenarios": args.scenarios,
        "model_families": args.model_families,
        "replications": args.replications,
        "n_paths": [args.train_paths, args.validation_paths, args.test_paths],
        "n_steps": args.steps,
        "window_size": args.window,
        "p_switch": args.p_switch,
        "trees": args.trees,
        "shuffle_repeats": args.shuffle_repeats,
        "block_size": args.block_size,
        "base_seed": args.base_seed,
        "representations": list(ORIGINAL_REPRESENTATIONS)
        + ["shuffled_signature", "block_shuffled_signature"],
        "signature_path": CausalFeatureExtractor.PATH_CONVENTION,
        "preprocessing": "RobustScaler plus fixed tail clipping, fit on training paths only",
        "selection": "hyperparameter selected by validation balanced accuracy",
        "test_policy": "test evaluated once after selection",
    }
    config_path = output_dir / "config.json"
    if args.resume and config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Cannot resume because config.json differs from current arguments.")
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    results_path = output_dir / "replications.csv"
    if args.resume and results_path.exists():
        rows = pd.read_csv(results_path).to_dict("records")
        completed = {
            (str(row["scenario"]), int(row["replication"])) for row in rows
        }
    else:
        rows = []
        completed = set()

    scenario_seeds = np.random.SeedSequence(args.base_seed).spawn(len(args.scenarios))
    started = time.monotonic()
    for scenario_index, scenario in enumerate(args.scenarios):
        base = int(scenario_seeds[scenario_index].generate_state(1, dtype=np.uint32)[0])
        seeds = replication_seeds(base, args.replications)
        for replication, seed in enumerate(seeds, start=1):
            if (scenario, replication) in completed:
                continue
            new_rows = evaluate_replication(
                scenario=scenario,
                replication=replication,
                seed=seed,
                n_paths=(args.train_paths, args.validation_paths, args.test_paths),
                n_steps=args.steps,
                window_size=args.window,
                p_switch=args.p_switch,
                trees=args.trees,
                model_families=args.model_families,
                shuffle_repeats=args.shuffle_repeats,
                block_size=args.block_size,
                jobs=args.jobs,
            )
            rows.extend(new_rows)
            current = pd.DataFrame(rows).sort_values(
                ["scenario", "replication", "model_family", "representation", "placebo_repeat"]
            )
            current.to_csv(results_path, index=False)
            if current.replication.nunique() >= 2:
                summary, contrasts = summarize(current)
                summary.to_csv(output_dir / "summary.csv", index=False)
                contrasts.to_csv(output_dir / "contrasts.csv", index=False)
            print(
                "{} replication {}/{} complete ({:.1f}s)".format(
                    scenario, replication, args.replications, time.monotonic() - started
                ),
                flush=True,
            )

    results = pd.DataFrame(rows).sort_values(
        ["scenario", "replication", "model_family", "representation", "placebo_repeat"]
    )
    results.to_csv(results_path, index=False)
    summary, contrasts = summarize(results)
    summary.to_csv(output_dir / "summary.csv", index=False)
    contrasts.to_csv(output_dir / "contrasts.csv", index=False)
    print("\nPrimary contrasts")
    print(contrasts.to_string(index=False, float_format=lambda value: "{:.5f}".format(value)))
    print("\nSaved to {}".format(output_dir))


if __name__ == "__main__":
    main()
