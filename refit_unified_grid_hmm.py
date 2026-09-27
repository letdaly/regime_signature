"""Recompute the HMM block of a finished unified-grid run from its archived paths.

The Markov-switching benchmark of ``experiment_unified_grid.py`` originally
used a single EM run from hmmlearn's default initialization, which in about
half of the Design A replications converged to a lower-likelihood optimum
with a short-lived outlier state.  ``fit_hmm`` now compares several
restarts by training log-likelihood.  This script refits the HMM block for
every archived replication (``paths/replication_NN.npz`` holds every
simulated path), rescoring only the ``hmm`` cell of each learner with the
grid's own ``score_cell``; every other cell is untouched, because the
other blocks do not depend on the HMM.  It rewrites ``replications.csv``,
``cells.csv``, ``summary.csv``, the ``hmm`` entries of the probability
archives and the ``hmm`` section of ``config.json``.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
import zlib
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import experiment_unified_grid as grid
from regime_pipeline import PathLevelSplits, SimulatedPath, WindowDataset


def load_archived_splits(path: Path, window_size: int) -> PathLevelSplits:
    """Rebuild path-level splits (labels and provenance, no features) from an archive."""

    archive = np.load(path)
    paths: Dict[str, list] = {"train": [], "validation": [], "test": []}
    for key in archive.files:
        if not key.endswith("__log_return"):
            continue
        split_name, path_id, _ = key.split("__")
        frame = pd.DataFrame(
            {
                "LogReturn": archive[key],
                "Regime": archive["{}__{}__regime".format(split_name, path_id)].astype(int),
            }
        )
        paths[split_name].append(
            SimulatedPath(path_id, frame, {}, zlib.crc32(path_id.encode("utf-8")))
        )
    for split_name in paths:
        paths[split_name].sort(key=lambda simulated: simulated.path_id)

    def dataset(split_name: str) -> WindowDataset:
        labels, path_ids, end_times = [], [], []
        for simulated in paths[split_name]:
            regimes = simulated.frame["Regime"].to_numpy(dtype=int)
            ends = np.arange(window_size - 1, len(regimes))
            labels.append(regimes[ends])
            end_times.append(ends)
            path_ids.append(np.repeat(simulated.path_id, len(ends)))
        return WindowDataset(
            X=np.zeros((sum(len(block) for block in labels), 0), dtype=float),
            y=np.concatenate(labels),
            path_ids=np.concatenate(path_ids),
            end_times=np.concatenate(end_times),
            feature_names=(),
        )

    splits = PathLevelSplits(
        train=dataset("train"),
        validation=dataset("validation"),
        test=dataset("test"),
        paths={split_name: tuple(split_paths) for split_name, split_paths in paths.items()},
    )
    splits.assert_disjoint()
    return splits


def refit_replication(task: Dict[str, object]) -> Dict[str, object]:
    started = time.monotonic()
    output_dir = Path(task["output_dir"])
    replication = int(task["replication"])
    seed = int(task["seed"])
    window_size = int(task["window_size"])
    learners = tuple(task["learners"])
    stem = "replication_{:02d}.npz".format(replication)

    splits = load_archived_splits(output_dir / "paths" / stem, window_size)
    splits, diagnostics = grid.add_hmm_block(splits, seed, window_size)
    columns = grid.block_columns(
        splits.train.feature_names + grid.STATISTICS_COLUMNS
    )["hmm"]
    updates: Dict[str, object] = dict(diagnostics)
    probabilities = {}
    for learner in learners:
        metrics, label, validation_probabilities, test_probabilities = grid.score_cell(
            splits, learner, "hmm", columns, seed, int(task["n_estimators"]), 1
        )
        updates.update(metrics)
        updates["{}_hmm_selected_setting".format(learner)] = label
        probabilities[learner] = (validation_probabilities, test_probabilities)

    archive_path = output_dir / "probabilities" / stem
    if archive_path.exists():
        arrays = dict(np.load(archive_path))
        for split_name in ("validation", "test"):
            if not np.array_equal(arrays[split_name + "_y"], getattr(splits, split_name).y):
                raise AssertionError("Archived labels differ from the rebuilt windows.")
        for learner, (validation_probabilities, test_probabilities) in probabilities.items():
            arrays["validation__{}__hmm".format(learner)] = grid.quantize(validation_probabilities)
            arrays["test__{}__hmm".format(learner)] = grid.quantize(test_probabilities)
        np.savez_compressed(archive_path, **arrays)
    updates["replication"] = replication
    updates["hmm_refit_seconds"] = float(time.monotonic() - started)
    return updates


def apply_updates(results: pd.DataFrame, updates: Dict[str, object]) -> None:
    index = results.index[results["replication"] == updates["replication"]]
    if len(index) != 1:
        raise ValueError("Replication {} is not unique in replications.csv.".format(updates["replication"]))
    for key, value in updates.items():
        results.loc[index, key] = value
    row = results.loc[index[0]].to_dict()
    learners = [learner for learner in grid.LEARNERS if "{}_hmm_test_ba".format(learner) in row]
    cells = [cell for cell in grid.CELLS if "{}_{}_test_ba".format(learners[0], cell) in row]
    for key, value in grid.contrasts(row, learners, cells).items():
        results.loc[index, key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", choices=tuple(grid.DESIGNS), required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = (
        Path("results/unified_grid") / args.design if args.output_dir is None else args.output_dir
    ).resolve()
    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text())
    results = pd.read_csv(output_dir / "replications.csv")
    if not (output_dir / "paths").exists():
        raise FileNotFoundError("No archived paths under {}.".format(output_dir))

    tasks = [
        {
            "output_dir": str(output_dir),
            "replication": int(row["replication"]),
            "seed": int(row["seed"]),
            "window_size": int(config["window_size"]),
            "n_estimators": int(config["n_estimators"]),
            "learners": tuple(config["learners"]),
        }
        for _, row in results.iterrows()
    ]
    started = time.monotonic()
    if args.jobs == 1:
        outcomes = [refit_replication(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(refit_replication, task) for task in tasks]
            outcomes = [future.result() for future in as_completed(futures)]
    for updates in outcomes:
        apply_updates(results, updates)
        print(
            "Replication {:2d}: restart {} loglik {:.1f} direct BA {:.4f} ({:.0f}s)".format(
                updates["replication"],
                int(updates["hmm_selected_restart"]),
                updates["hmm_log_likelihood"],
                updates["hmm_direct_test_ba"],
                updates["hmm_refit_seconds"],
            ),
            flush=True,
        )
    results = results.sort_values("replication").reset_index(drop=True)
    grid.save_tables(results, output_dir)
    config["hmm"].update(
        {
            "restarts": grid.HMM_RESTARTS,
            "persistent_initialization_diagonal": grid.HMM_PERSISTENT_DIAGONAL,
            "selection": "highest training log-likelihood",
            "refit_from_archived_paths": True,
        }
    )
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    summary = grid.summarize_results(results).set_index("metric")
    print(
        "\nHMM direct test BA {:.4f} [{:.4f}, {:.4f}]; restarts selected: {}".format(
            summary.loc["BA_hmm_direct", "mean"],
            summary.loc["BA_hmm_direct", "ci_low"],
            summary.loc["BA_hmm_direct", "ci_high"],
            results["hmm_selected_restart"].astype(int).value_counts().sort_index().to_dict(),
        )
    )
    print("Total time: {:.1f}s".format(time.monotonic() - started))


if __name__ == "__main__":
    main()
