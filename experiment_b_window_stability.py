"""Window-stability check for Signature's fast-kappa recall advantage.

The data-generating process is held fixed across windows: every window uses the
same replication seeds and the same minimum regime dwell.  Regime 2 is the
fast-kappa class (kappa in [14, 18]).
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import ttest_1samp

from experiment_a_monte_carlo import (
    add_path_count_arguments,
    add_signature_kind_argument,
    guard_output_directory,
    replication_seeds,
)
from experiment_b_monte_carlo import evaluate_paired_replication


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=int, nargs="+", default=[25, 50, 100, 200])
    parser.add_argument("--replications", type=int, default=50)
    add_path_count_arguments(parser)
    add_signature_kind_argument(parser)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--p-switch", type=float, default=0.002)
    parser.add_argument("--min-dwell", type=int, default=50)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument(
        "--base-seed",
        type=int,
        default=20260829,
        help="Independent follow-up seed; must not reproduce the primary B seeds.",
    )
    parser.add_argument(
        "--primary-config",
        type=Path,
        default=Path("results/experiment_b_diagnostics/config.json"),
        help="Primary Experiment B config whose replication seeds are embargoed.",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/main_grid/heston_matched_marginal_window_stability")
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _task(task: Dict[str, object]) -> Dict[str, float]:
    started = time.monotonic()
    window = int(task.pop("window"))
    row = evaluate_paired_replication(**task)
    row["window_size"] = window
    row["elapsed_seconds"] = float(time.monotonic() - started)
    return row


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for window, group in results.groupby("window_size", sort=True):
        delta = group["signature_minus_statistics_recall_2"].dropna().to_numpy()
        n = len(delta)
        mean = float(np.mean(delta))
        standard_error = float(np.std(delta, ddof=1) / np.sqrt(n))
        critical = float(student_t.ppf(0.975, n - 1))
        p_value = float(ttest_1samp(delta, 0.0, alternative="greater").pvalue)
        rows.append(
            {
                "window_size": int(window),
                "n": n,
                "statistics_fast_recall_mean": float(group["statistics_test_recall_2"].mean()),
                "signature_fast_recall_mean": float(group["signature_test_recall_2"].mean()),
                "delta_fast_recall_mean": mean,
                "ci_low": mean - critical * standard_error,
                "ci_high": mean + critical * standard_error,
                "signature_win_rate": float(np.mean(delta > 0.0)),
                "one_sided_p_value": p_value,
            }
        )
    summary = pd.DataFrame(rows)
    summary["holm_adjusted_p_value"] = _holm_adjust(
        summary["one_sided_p_value"].to_numpy()
    )
    summary["positive_95_ci"] = summary["ci_low"] > 0.0
    summary["positive_after_holm_0.05"] = summary["holm_adjusted_p_value"] < 0.05
    return summary


def summarize_tradeoffs(results: pd.DataFrame) -> pd.DataFrame:
    """Show whether fast-kappa recall is purchased by losses on other metrics."""

    rows: List[Dict[str, float]] = []
    for window, group in results.groupby("window_size", sort=True):
        for metric in ("recall_0", "recall_1", "recall_2", "ba", "macro_f1"):
            statistics = group["statistics_test_" + metric]
            signature = group["signature_test_" + metric]
            delta = (signature - statistics).dropna().to_numpy()
            n = len(delta)
            mean = float(np.mean(delta))
            standard_error = float(np.std(delta, ddof=1) / np.sqrt(n))
            critical = float(student_t.ppf(0.975, n - 1))
            rows.append(
                {
                    "window_size": int(window),
                    "metric": metric,
                    "n": n,
                    "statistics_mean": float(statistics.mean()),
                    "signature_mean": float(signature.mean()),
                    "delta_mean": mean,
                    "ci_low": mean - critical * standard_error,
                    "ci_high": mean + critical * standard_error,
                    "signature_win_rate": float(np.mean(delta > 0.0)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.replications < 2 or args.jobs < 1:
        raise ValueError("replications must be >= 2 and jobs must be positive.")
    if any(window < 3 or window > args.steps for window in args.windows):
        raise ValueError("Every window must lie between 3 and steps.")
    if args.min_dwell < 1:
        raise ValueError("min-dwell must be positive.")

    seeds = replication_seeds(args.base_seed, args.replications)
    primary_config_path = args.primary_config.resolve()
    if primary_config_path.exists():
        primary_config = json.loads(primary_config_path.read_text())
        primary_seeds = set(
            replication_seeds(
                int(primary_config["base_seed"]),
                int(primary_config["replications"]),
            )
        )
        overlap = primary_seeds.intersection(seeds)
        if overlap:
            raise ValueError(
                "Window follow-up reuses {} primary Experiment B replication "
                "seed(s). Choose an independent --base-seed.".format(len(overlap))
            )

    output_dir = args.output_dir.resolve()
    guard_output_directory(output_dir, args.resume)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "scenario": "matched_stationary_cir_marginal_different_kappa",
        "purpose": "fast_kappa_signature_window_stability",
        "windows": sorted(set(args.windows)),
        "replications": args.replications,
        "n_paths": [args.train_paths, args.validation_paths, args.test_paths],
        "n_steps": args.steps,
        "p_switch": args.p_switch,
        "min_dwell": args.min_dwell,
        "n_estimators": args.trees,
        "signature_kind": args.signature_kind,
        "base_seed": args.base_seed,
        "primary_config": str(primary_config_path),
        "seed_overlap_with_primary": 0,
        "representations": ["statistics", "signature"],
        "fast_kappa_regime": 2,
        "fast_kappa_range": [14.0, 18.0],
    }
    config_path = output_dir / "config.json"
    if args.resume and config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Cannot resume: current arguments differ from config.json.")
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    results_path = output_dir / "replications.csv"
    if args.resume and results_path.exists():
        rows = pd.read_csv(results_path).to_dict("records")
        completed = {(int(row["window_size"]), int(row["replication"])) for row in rows}
    else:
        rows = []
        completed = set()

    tasks = []
    for window in sorted(set(args.windows)):
        for replication, seed in enumerate(seeds, start=1):
            if (window, replication) in completed:
                continue
            tasks.append(
                {
                    "window": window,
                    "replication": replication,
                    "seed": seed,
                    "n_paths": (args.train_paths, args.validation_paths, args.test_paths),
                    "n_steps": args.steps,
                    "window_size": window,
                    "p_switch": args.p_switch,
                    "n_estimators": args.trees,
                    "event_tolerance": 100,
                    "model_jobs": 1,
                    "min_dwell": args.min_dwell,
                    "representations": ("statistics", "signature"),
                    "evaluate_detector": False,
                    "signature_kind": args.signature_kind,
                }
            )

    def checkpoint(row: Dict[str, float]) -> None:
        rows.append(row)
        current = pd.DataFrame(rows).sort_values(["window_size", "replication"])
        current.to_csv(results_path, index=False)
        if current.groupby("window_size").size().min() >= 2:
            summarize(current).to_csv(output_dir / "summary.csv", index=False)
        print(
            "window={} replication={}/{} fast-recall(stat/sig)={:.3f}/{:.3f} delta={:+.3f} ({:.1f}s)".format(
                int(row["window_size"]),
                int(row["replication"]),
                args.replications,
                row["statistics_test_recall_2"],
                row["signature_test_recall_2"],
                row["signature_minus_statistics_recall_2"],
                row["elapsed_seconds"],
            ),
            flush=True,
        )

    if args.jobs == 1:
        for task in tasks:
            checkpoint(_task(task))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(_task, task) for task in tasks]
            for future in as_completed(futures):
                checkpoint(future.result())

    results = pd.DataFrame(rows).sort_values(["window_size", "replication"])
    results.to_csv(results_path, index=False)
    summary = summarize(results)
    summary.to_csv(output_dir / "summary.csv", index=False)
    summarize_tradeoffs(results).to_csv(output_dir / "tradeoffs.csv", index=False)
    print("\nFast-kappa window-stability summary")
    print(summary.to_string(index=False, float_format=lambda value: "{:.6f}".format(value)))
    print("\nSaved results to {}".format(output_dir))


if __name__ == "__main__":
    main()
