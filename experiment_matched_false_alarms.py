"""Regime-change detection at matched false-alarm rates.

Every detector is calibrated on the validation paths of a unified-grid
replication to a common false-alarm budget and then applied once to its test
paths, so detection probability and delay are compared at the same cost
rather than at separately selected operating points.

Detectors
---------
* Classifier detectors: the archived validation and test probabilities of the
  seven Table-1 cells for every learner (``results/unified_grid/<design>/
  probabilities``) are passed through the causal detector of
  ``regime_detection`` (exponential smoothing with rate ``alpha``, a switch
  declared when another state's smoothed probability stays at or above
  ``threshold`` for ``confirmations`` observations).
* HMM filter: the forward-filtered state probabilities of the grid's
  three-state Gaussian HMM (refitted from the archived paths with the grid's
  own code) through the same detector.
* CUSUM: Page's two-sided cumulative sum on log squared returns relative to
  an exponentially weighted reference level, with allowance ``k``, reference
  half-life and decision limit ``h``; on an alarm the sums reset and the
  reference re-anchors to the mean of the most recent observations.

Calibration
-----------
For each target rate tau in {0.5, 1, 2, 5} false alarms per 1,000
observations, every detector's setting (all smoothing rates and confirmation
counts with a fine threshold grid; all CUSUM allowances and half-lives with a
geometric limit grid) is scored on the validation paths and the setting with
the highest validation detection probability among those whose validation
false-alarm rate does not exceed tau is kept (the lowest-rate setting when
none qualifies).  Realized test false-alarm rate, detection probability
within ``tolerance`` observations, delay and state accuracy are then
reported.

Matching is state-agnostic: an alarm matches the earliest unmatched true
change that precedes it by at most ``tolerance`` observations, whatever state
the detector announces; unmatched alarms are false alarms.  Classifier and
HMM detectors additionally report the share of changes whose first matched
alarm also named the new state (``state_detection_probability``).  All
detectors are scored on the same observations: window endpoints from
``window_size - 1`` on, so that the CUSUM and the HMM see the same 50-return
warm-up as the classifiers.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import lfilter

import experiment_unified_grid as grid
from experiment_a_monte_carlo import guard_output_directory, mean_t_interval
from refit_unified_grid_hmm import load_archived_splits
from regime_detection import REGIME_IDS, causal_detector
from regime_pipeline import PathLevelSplits


TARGETS = (0.5, 1.0, 2.0, 5.0)
DEFAULT_TOLERANCE = 100
ALPHAS = (0.2, 0.5, 1.0)
CONFIRMATIONS = (1, 3, 5)
THRESHOLDS = tuple(
    float(value)
    for value in np.round(
        np.concatenate((np.arange(0.34, 0.90, 0.04), np.arange(0.90, 0.995, 0.01))), 3
    )
)
CUSUM_ALLOWANCES = (0.25, 0.5, 1.0)
CUSUM_HALF_LIVES = (50, 250)
CUSUM_LIMITS = tuple(float(value) for value in np.round(np.geomspace(2.0, 200.0, 24), 3))
CUSUM_BURN_IN = 10
CUSUM_FLOOR = 1e-12
REPRESENTATIONS = (
    "statistics",
    "statistics_logsig",
    "statistics_rawsig",
    "b3",
    "b3_logsig",
    "b3_rawsig",
)
BASELINE_OF = {
    "statistics_logsig": "statistics",
    "statistics_rawsig": "statistics",
    "b3_logsig": "b3",
    "b3_rawsig": "b3",
}
METRICS = (
    "false_alarms_per_1000",
    "detection_probability",
    "chance_detection_probability",
    "mean_delay",
    "median_delay",
    "state_detection_probability",
    "detector_balanced_accuracy",
)


# ---------------------------------------------------------------------------
# Detector primitives
# ---------------------------------------------------------------------------


def smooth_probabilities(probabilities: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential smoothing identical to ``causal_detector``'s recursion."""

    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must lie in (0, 1].")
    if alpha == 1.0:
        return np.asarray(probabilities, dtype=float)
    initial = ((1.0 - alpha) * probabilities[0])[None, :]
    smoothed, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], probabilities, axis=0, zi=initial)
    return smoothed


def detector_alarms(
    proposed: Sequence[int],
    confidence: Sequence[float],
    end_times: Sequence[int],
    threshold: float,
    confirmations: int,
) -> List[Tuple[int, int]]:
    """Alarm times and announced states of ``causal_detector`` from precomputed inputs."""

    current = int(proposed[0])
    candidate = current
    count = 0
    alarms: List[Tuple[int, int]] = []
    for index in range(1, len(proposed)):
        state = proposed[index]
        if state != current and confidence[index] >= threshold:
            if state == candidate:
                count += 1
            else:
                candidate = state
                count = 1
            if count >= confirmations:
                current = candidate
                count = 0
                alarms.append((int(end_times[index]), current))
        else:
            candidate = current
            count = 0
    return alarms


def decisions_from_alarms(
    initial_state: int, alarms: Sequence[Tuple[int, int]], end_times: np.ndarray
) -> np.ndarray:
    decisions = np.full(len(end_times), int(initial_state), dtype=int)
    for time_index, state in alarms:
        decisions[end_times >= time_index] = state
    return decisions


def change_times(states: np.ndarray, end_times: np.ndarray) -> List[Tuple[int, int]]:
    indices = np.flatnonzero(states[1:] != states[:-1]) + 1
    return [(int(end_times[index]), int(states[index])) for index in indices]


def match_alarms(
    true_events: Sequence[Tuple[int, int]],
    alarms: Sequence[Tuple[int, int]],
    tolerance: int,
) -> Tuple[List[int], List[bool], int]:
    """State-agnostic greedy matching; returns delays, state-correct flags, false alarms."""

    used = set()
    delays: List[int] = []
    state_correct: List[bool] = []
    for true_time, new_state in true_events:
        for index, (alarm_time, alarm_state) in enumerate(alarms):
            if index in used or alarm_time < true_time:
                continue
            if alarm_time > true_time + tolerance:
                break
            used.add(index)
            delays.append(alarm_time - true_time)
            state_correct.append(alarm_state == new_state)
            break
    return delays, state_correct, len(alarms) - len(used)


class EventTally:
    """Accumulates matched changes, delays and false alarms over paths."""

    def __init__(self) -> None:
        self.n_true = 0
        self.n_observations = 0
        self.n_false = 0
        self.delays: List[int] = []
        self.state_correct: List[bool] = []

    def add(
        self,
        true_events: Sequence[Tuple[int, int]],
        alarms: Sequence[Tuple[int, int]],
        n_observations: int,
        tolerance: int,
    ) -> None:
        delays, state_correct, false = match_alarms(true_events, alarms, tolerance)
        self.n_true += len(true_events)
        self.n_observations += n_observations
        self.n_false += false
        self.delays.extend(delays)
        self.state_correct.extend(state_correct)

    def metrics(self, state_aware: bool, tolerance: int) -> Dict[str, float]:
        """Rates, delays and the chance level of random alarms at the same total rate.

        ``chance_detection_probability`` is 1 - exp(-(tolerance + 1) * rate) with
        ``rate`` the detector's total alarm rate per observation: the probability
        that a memoryless alarm process firing that often lands at least one
        alarm inside a change's tolerance window.
        """

        matched = len(self.delays)
        total_rate = (self.n_false + matched) / self.n_observations
        return {
            "n_observations": self.n_observations,
            "false_alarms_per_1000": 1000.0 * self.n_false / self.n_observations,
            "chance_detection_probability": float(1.0 - np.exp(-(tolerance + 1) * total_rate)),
            "detection_probability": matched / self.n_true if self.n_true else np.nan,
            "mean_delay": float(np.mean(self.delays)) if self.delays else np.nan,
            "median_delay": float(np.median(self.delays)) if self.delays else np.nan,
            "state_detection_probability": (
                float(np.sum(self.state_correct)) / self.n_true
                if state_aware and self.n_true
                else np.nan
            ),
            "n_true_changes": self.n_true,
            "n_matched": matched,
            "n_false_alarms": self.n_false,
        }


def balanced_accuracy(truth: np.ndarray, decisions: np.ndarray) -> float:
    recalls = []
    for regime in REGIME_IDS:
        mask = truth == regime
        if mask.any():
            recalls.append(float(np.mean(decisions[mask] == regime)))
    return float(np.mean(recalls))


# ---------------------------------------------------------------------------
# Path-level inputs
# ---------------------------------------------------------------------------


class PathBlock:
    """One path's window labels, end times and per-detector inputs."""

    def __init__(self, truth: np.ndarray, end_times: np.ndarray) -> None:
        self.truth = truth
        self.end_times = end_times
        self.true_events = change_times(truth, end_times)
        self.n_observations = len(truth)


def split_paths(y: np.ndarray, path_ids: np.ndarray, end_times: np.ndarray) -> Dict[str, np.ndarray]:
    masks = {}
    for path_id in dict.fromkeys(path_ids.tolist()):
        masks[path_id] = path_ids == path_id
    return masks


# ---------------------------------------------------------------------------
# Classifier / HMM detector: sweep and evaluation
# ---------------------------------------------------------------------------


def probability_sweep(
    probabilities: np.ndarray,
    y: np.ndarray,
    masks: Dict[str, np.ndarray],
    end_times: np.ndarray,
    tolerance: int,
) -> pd.DataFrame:
    """Validation metrics of every (alpha, confirmations, threshold) setting."""

    blocks = {path_id: PathBlock(y[mask], end_times[mask]) for path_id, mask in masks.items()}
    rows = []
    for alpha in ALPHAS:
        per_path = {}
        for path_id, mask in masks.items():
            smoothed = smooth_probabilities(probabilities[mask], alpha)
            proposed = np.argmax(smoothed, axis=1)
            confidence = smoothed[np.arange(len(proposed)), proposed]
            per_path[path_id] = (proposed.tolist(), confidence.tolist(), blocks[path_id].end_times.tolist())
        for confirmations in CONFIRMATIONS:
            for threshold in THRESHOLDS:
                tally = EventTally()
                for path_id, (proposed, confidence, times) in per_path.items():
                    alarms = detector_alarms(proposed, confidence, times, threshold, confirmations)
                    block = blocks[path_id]
                    tally.add(block.true_events, alarms, block.n_observations, tolerance)
                rows.append(
                    {
                        "alpha": alpha,
                        "confirmations": confirmations,
                        "threshold": threshold,
                        **tally.metrics(state_aware=True, tolerance=tolerance),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_probability_detector(
    probabilities: np.ndarray,
    y: np.ndarray,
    masks: Dict[str, np.ndarray],
    end_times: np.ndarray,
    alpha: float,
    confirmations: int,
    threshold: float,
    tolerance: int,
) -> Dict[str, float]:
    tally = EventTally()
    all_truth, all_decisions = [], []
    for path_id, mask in masks.items():
        block = PathBlock(y[mask], end_times[mask])
        smoothed = smooth_probabilities(probabilities[mask], alpha)
        proposed = np.argmax(smoothed, axis=1)
        confidence = smoothed[np.arange(len(proposed)), proposed]
        alarms = detector_alarms(
            proposed.tolist(), confidence.tolist(), block.end_times.tolist(), threshold, confirmations
        )
        tally.add(block.true_events, alarms, block.n_observations, tolerance)
        all_truth.append(block.truth)
        all_decisions.append(decisions_from_alarms(int(proposed[0]), alarms, block.end_times))
    metrics = tally.metrics(state_aware=True, tolerance=tolerance)
    metrics["detector_balanced_accuracy"] = balanced_accuracy(
        np.concatenate(all_truth), np.concatenate(all_decisions)
    )
    return metrics


# ---------------------------------------------------------------------------
# CUSUM detector
# ---------------------------------------------------------------------------


def log_squared_returns(returns: np.ndarray) -> np.ndarray:
    return np.log(np.asarray(returns, dtype=float) ** 2 + CUSUM_FLOOR)


def cusum_alarms(
    x: Sequence[float],
    start_index: int,
    allowance: float,
    half_life: float,
    limit: float,
    burn_in: int = CUSUM_BURN_IN,
) -> List[Tuple[int, int]]:
    """Two-sided CUSUM alarms (time, +1 up / -1 down) at or after ``start_index``."""

    if burn_in < 1 or limit <= 0 or allowance < 0 or half_life <= 0:
        raise ValueError("Invalid CUSUM setting.")
    rate = 1.0 - 0.5 ** (1.0 / half_life)
    reference = float(np.mean(x[:burn_in]))
    upper = lower = 0.0
    alarms: List[Tuple[int, int]] = []
    for index in range(burn_in, len(x)):
        deviation = x[index] - reference
        upper = max(0.0, upper + deviation - allowance)
        lower = max(0.0, lower - deviation - allowance)
        if upper > limit or lower > limit:
            if index >= start_index:
                alarms.append((index, 1 if upper > limit else -1))
            upper = lower = 0.0
            reference = float(np.mean(x[max(0, index - burn_in + 1) : index + 1]))
        else:
            reference = rate * x[index] + (1.0 - rate) * reference
    return alarms


def cusum_sweep(
    paths: Sequence, window_size: int, tolerance: int
) -> pd.DataFrame:
    blocks = []
    for simulated in paths:
        regimes = simulated.frame["Regime"].to_numpy(dtype=int)
        end_times = np.arange(window_size - 1, len(regimes))
        blocks.append(
            (
                log_squared_returns(simulated.frame["LogReturn"].to_numpy()).tolist(),
                PathBlock(regimes[end_times], end_times),
            )
        )
    rows = []
    for allowance in CUSUM_ALLOWANCES:
        for half_life in CUSUM_HALF_LIVES:
            for limit in CUSUM_LIMITS:
                tally = EventTally()
                for x, block in blocks:
                    alarms = cusum_alarms(x, window_size - 1, allowance, half_life, limit)
                    tally.add(block.true_events, alarms, block.n_observations, tolerance)
                rows.append(
                    {
                        "allowance": allowance,
                        "half_life": half_life,
                        "limit": limit,
                        **tally.metrics(state_aware=False, tolerance=tolerance),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_cusum(
    paths: Sequence, window_size: int, allowance: float, half_life: float, limit: float, tolerance: int
) -> Dict[str, float]:
    tally = EventTally()
    for simulated in paths:
        regimes = simulated.frame["Regime"].to_numpy(dtype=int)
        end_times = np.arange(window_size - 1, len(regimes))
        block = PathBlock(regimes[end_times], end_times)
        x = log_squared_returns(simulated.frame["LogReturn"].to_numpy()).tolist()
        alarms = cusum_alarms(x, window_size - 1, allowance, half_life, limit)
        tally.add(block.true_events, alarms, block.n_observations, tolerance)
    metrics = tally.metrics(state_aware=False, tolerance=tolerance)
    metrics["detector_balanced_accuracy"] = np.nan
    return metrics


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def select_setting(sweep: pd.DataFrame, target: float) -> Tuple[pd.Series, bool]:
    """Highest validation detection probability with false alarms <= target."""

    feasible = sweep[sweep["false_alarms_per_1000"] <= target]
    if len(feasible):
        ordered = feasible.sort_values(
            ["detection_probability", "false_alarms_per_1000", "mean_delay"],
            ascending=[False, True, True],
            na_position="last",
        )
        return ordered.iloc[0], True
    ordered = sweep.sort_values(
        ["false_alarms_per_1000", "detection_probability"], ascending=[True, False]
    )
    return ordered.iloc[0], False


# ---------------------------------------------------------------------------
# Replication
# ---------------------------------------------------------------------------


def hmm_filter_probabilities(splits: PathLevelSplits, seed: int, window_size: int) -> Dict[str, np.ndarray]:
    matched, _ = grid.hmm_block(splits, seed, window_size)
    return matched


def evaluate_replication(task: Dict[str, object]) -> List[Dict[str, object]]:
    started = time.monotonic()
    design = str(task["design"])
    replication = int(task["replication"])
    seed = int(task["seed"])
    window_size = int(task["window_size"])
    tolerance = int(task["tolerance"])
    learners = tuple(task["learners"])
    cells = tuple(task["cells"])
    grid_dir = Path(task["grid_dir"])
    stem = "replication_{:02d}.npz".format(replication)

    archive = np.load(grid_dir / "probabilities" / stem)
    splits = load_archived_splits(grid_dir / "paths" / stem, window_size)
    data = {}
    for split_name in ("validation", "test"):
        y = archive[split_name + "_y"]
        path_ids = archive[split_name + "_path_ids"]
        end_times = archive[split_name + "_end_times"]
        dataset = getattr(splits, split_name)
        if not (np.array_equal(y, dataset.y) and np.array_equal(end_times, dataset.end_times)):
            raise AssertionError("Archived windows do not match the rebuilt paths.")
        data[split_name] = (y, split_paths(y, path_ids, end_times), end_times)

    detectors: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    for learner in learners:
        for cell in cells:
            detectors[(cell, learner)] = {
                split_name: grid.dequantize(archive["{}__{}__{}".format(split_name, learner, cell)])
                for split_name in ("validation", "test")
            }
    hmm = hmm_filter_probabilities(splits, seed, window_size)
    detectors[("hmm_filter", "none")] = {"validation": hmm["validation"], "test": hmm["test"]}

    rows: List[Dict[str, object]] = []
    base = {"design": design, "replication": replication, "seed": seed}
    for (cell, learner), probabilities in detectors.items():
        y_val, masks_val, times_val = data["validation"]
        y_test, masks_test, times_test = data["test"]
        sweep = probability_sweep(probabilities["validation"], y_val, masks_val, times_val, tolerance)
        for target in TARGETS:
            chosen, feasible = select_setting(sweep, target)
            test = evaluate_probability_detector(
                probabilities["test"],
                y_test,
                masks_test,
                times_test,
                float(chosen["alpha"]),
                int(chosen["confirmations"]),
                float(chosen["threshold"]),
                tolerance,
            )
            rows.append(
                {
                    **base,
                    "detector": cell,
                    "learner": learner,
                    "target": target,
                    "feasible": bool(feasible),
                    "alpha": float(chosen["alpha"]),
                    "confirmations": int(chosen["confirmations"]),
                    "threshold": float(chosen["threshold"]),
                    "validation_false_alarms_per_1000": float(chosen["false_alarms_per_1000"]),
                    "validation_detection_probability": float(chosen["detection_probability"]),
                    **{"test_" + key: value for key, value in test.items()},
                }
            )

    sweep = cusum_sweep(splits.paths["validation"], window_size, tolerance)
    for target in TARGETS:
        chosen, feasible = select_setting(sweep, target)
        test = evaluate_cusum(
            splits.paths["test"],
            window_size,
            float(chosen["allowance"]),
            float(chosen["half_life"]),
            float(chosen["limit"]),
            tolerance,
        )
        rows.append(
            {
                **base,
                "detector": "cusum",
                "learner": "none",
                "target": target,
                "feasible": bool(feasible),
                "cusum_allowance": float(chosen["allowance"]),
                "cusum_half_life": float(chosen["half_life"]),
                "cusum_limit": float(chosen["limit"]),
                "validation_false_alarms_per_1000": float(chosen["false_alarms_per_1000"]),
                "validation_detection_probability": float(chosen["detection_probability"]),
                **{"test_" + key: value for key, value in test.items()},
            }
        )
    elapsed = time.monotonic() - started
    for row in rows:
        row["elapsed_seconds"] = elapsed
    return rows


# ---------------------------------------------------------------------------
# Tables and figure
# ---------------------------------------------------------------------------


def attach_selected_learner(results: pd.DataFrame, grid_results: pd.DataFrame) -> pd.DataFrame:
    """Add rows for the validation-BA-selected learner of each representation."""

    validation = grid_results.set_index("replication")
    selected = []
    for replication in results["replication"].unique():
        for cell in REPRESENTATIONS:
            scores = {
                learner: validation.loc[replication, "{}_{}_validation_ba".format(learner, cell)]
                for learner in grid.LEARNERS
                if "{}_{}_validation_ba".format(learner, cell) in validation.columns
            }
            best = max(scores, key=scores.get)
            block = results[
                (results["replication"] == replication)
                & (results["detector"] == cell)
                & (results["learner"] == best)
            ].copy()
            block["learner"] = "selected"
            block["selected_learner"] = best
            selected.append(block)
    return pd.concat([results] + selected, ignore_index=True)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = results["replication"].nunique()
    for (detector, learner, target), group in results.groupby(["detector", "learner", "target"]):
        record = {"detector": detector, "learner": learner, "target": target, "n": n,
                  "feasible_share": float(group["feasible"].mean())}
        for metric in METRICS:
            values = group["test_" + metric].dropna()
            if len(values) >= 2:
                interval = mean_t_interval(values)
                record[metric] = interval["mean"]
                record[metric + "_ci_low"] = interval["ci_low"]
                record[metric + "_ci_high"] = interval["ci_high"]
            else:
                record[metric] = record[metric + "_ci_low"] = record[metric + "_ci_high"] = np.nan
        record["validation_false_alarms_per_1000"] = float(group["validation_false_alarms_per_1000"].mean())
        rows.append(record)
    return pd.DataFrame(rows)


def paired_contrasts(results: pd.DataFrame) -> pd.DataFrame:
    """Augmented minus base, and representation minus CUSUM / HMM, per learner and target."""

    indexed = results.set_index(["detector", "learner", "target", "replication"]).sort_index()
    rows = []
    learners = sorted(results["learner"].unique())

    def add(detector, learner, other_detector, other_learner, target, label):
        try:
            left = indexed.loc[(detector, learner, target)]
            right = indexed.loc[(other_detector, other_learner, target)]
        except KeyError:
            return
        joined = left.join(right, lsuffix="_a", rsuffix="_b", how="inner")
        record = {"contrast": label, "detector": detector, "learner": learner,
                  "baseline": other_detector, "target": target, "n": len(joined)}
        for metric in ("detection_probability", "mean_delay", "false_alarms_per_1000",
                       "state_detection_probability", "detector_balanced_accuracy"):
            diff = (joined["test_" + metric + "_a"] - joined["test_" + metric + "_b"]).dropna()
            if len(diff) >= 2:
                interval = mean_t_interval(diff)
                record["delta_" + metric] = interval["mean"]
                record["delta_" + metric + "_ci_low"] = interval["ci_low"]
                record["delta_" + metric + "_ci_high"] = interval["ci_high"]
                record["delta_" + metric + "_positive"] = int((diff > 0).sum())
        rows.append(record)

    for target in TARGETS:
        for (detector, learner), group in results[results["target"] == target].groupby(["detector", "learner"]):
            diff = (group["test_detection_probability"] - group["test_chance_detection_probability"]).dropna()
            if len(diff) >= 2:
                interval = mean_t_interval(diff)
                rows.append(
                    {
                        "contrast": "{} minus chance".format(detector),
                        "detector": detector,
                        "learner": learner,
                        "baseline": "chance",
                        "target": target,
                        "n": len(diff),
                        "delta_detection_probability": interval["mean"],
                        "delta_detection_probability_ci_low": interval["ci_low"],
                        "delta_detection_probability_ci_high": interval["ci_high"],
                        "delta_detection_probability_positive": int((diff > 0).sum()),
                    }
                )
        for learner in learners:
            if learner == "none":
                continue
            for cell, baseline in BASELINE_OF.items():
                add(cell, learner, baseline, learner, target, "{} minus {}".format(cell, baseline))
            for cell in REPRESENTATIONS:
                add(cell, learner, "cusum", "none", target, "{} minus CUSUM".format(cell))
                add(cell, learner, "hmm_filter", "none", target, "{} minus HMM filter".format(cell))
        add("hmm_filter", "none", "cusum", "none", target, "HMM filter minus CUSUM")
    return pd.DataFrame(rows)


def make_figure(summaries: Dict[str, pd.DataFrame], path: Path, tolerance: int = DEFAULT_TOLERANCE) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    styles = {
        ("cusum", "none"): ("CUSUM (log squared returns)", "black", "--", "x"),
        ("hmm_filter", "none"): ("HMM filter", "0.45", "-.", "^"),
        ("statistics", "selected"): ("Statistics", "#1f77b4", "-", "o"),
        ("statistics_logsig", "selected"): ("Statistics + log-sig", "#2ca02c", "-", "s"),
        ("statistics_rawsig", "selected"): ("Statistics + raw-sig", "#d62728", "-", "D"),
        ("b3", "selected"): ("B3", "#1f77b4", ":", "o"),
        ("b3_logsig", "selected"): ("B3 + log-sig", "#2ca02c", ":", "s"),
        ("b3_rawsig", "selected"): ("B3 + raw-sig", "#d62728", ":", "D"),
    }
    titles = {"scale_only": "Design A (scale)", "matched_marginal": "Design B (speed)"}
    fig, axes = plt.subplots(1, len(summaries), figsize=(5.2 * len(summaries), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, (design, summary) in zip(axes, summaries.items()):
        for (detector, learner), (label, color, linestyle, marker) in styles.items():
            block = summary[(summary["detector"] == detector) & (summary["learner"] == learner)].sort_values("target")
            if block.empty:
                continue
            x = block["false_alarms_per_1000"].to_numpy()
            y = block["detection_probability"].to_numpy()
            axis.errorbar(
                x,
                y,
                yerr=[y - block["detection_probability_ci_low"], block["detection_probability_ci_high"] - y],
                xerr=[x - block["false_alarms_per_1000_ci_low"], block["false_alarms_per_1000_ci_high"] - x],
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=4,
                linewidth=1.2,
                capsize=2,
                label=label,
            )
        chance = summary[summary["learner"].isin(["selected", "none"])].groupby("target")[
            ["false_alarms_per_1000", "chance_detection_probability"]
        ].mean().sort_values("false_alarms_per_1000")
        axis.plot(
            chance["false_alarms_per_1000"],
            chance["chance_detection_probability"],
            color="0.6",
            linestyle=":",
            linewidth=1.0,
            label="random alarms at the same rate",
        )
        axis.set_xscale("log")
        axis.set_xticks(TARGETS)
        axis.set_xticklabels(["{:g}".format(target) for target in TARGETS])
        axis.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        for target in TARGETS:
            axis.axvline(target, color="0.85", linewidth=0.6, zorder=0)
        axis.set_xlabel("Realized false alarms per 1,000 test observations")
        axis.set_title(titles.get(design, design))
        axis.grid(True, which="both", linewidth=0.3, alpha=0.5)
    axes[0].set_ylabel("Detection probability within {} observations".format(tolerance))
    axes[0].set_ylim(0, 1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_design(design: str, grid_root: Path, output_root: Path, jobs: int, tolerance: int, resume: bool) -> pd.DataFrame:
    grid_dir = grid_root / design
    config = json.loads((grid_dir / "config.json").read_text())
    grid_results = pd.read_csv(grid_dir / "replications.csv")
    output_dir = output_root / design
    guard_output_directory(output_dir, resume)
    output_dir.mkdir(parents=True, exist_ok=True)
    replications_path = output_dir / "replications.csv"
    if resume and replications_path.exists():
        existing = pd.read_csv(replications_path)
        existing = existing[existing["learner"] != "selected"]
        done = set(existing["replication"].astype(int))
        rows = existing.to_dict("records")
    else:
        done = set()
        rows = []
    tasks = [
        {
            "design": design,
            "replication": int(row["replication"]),
            "seed": int(row["seed"]),
            "window_size": int(config["window_size"]),
            "tolerance": tolerance,
            "learners": tuple(config["learners"]),
            "cells": tuple(config["table_cells"]),
            "grid_dir": str(grid_dir),
        }
        for _, row in grid_results.iterrows()
        if int(row["replication"]) not in done
    ]
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "design": design,
                "grid_dir": str(grid_dir),
                "grid_config": config,
                "targets_per_1000": list(TARGETS),
                "tolerance": tolerance,
                "alphas": list(ALPHAS),
                "confirmations": list(CONFIRMATIONS),
                "thresholds": list(THRESHOLDS),
                "cusum": {
                    "input": "log(r^2 + 1e-12)",
                    "allowances": list(CUSUM_ALLOWANCES),
                    "reference_half_lives": list(CUSUM_HALF_LIVES),
                    "limits": list(CUSUM_LIMITS),
                    "burn_in": CUSUM_BURN_IN,
                    "restart": "sums reset and reference re-anchored to the mean of the last burn_in observations",
                },
                "selection": "max validation detection probability subject to validation false alarms <= target; lowest-rate setting if none qualifies",
                "matching": "state-agnostic within tolerance; state_detection_probability additionally requires the announced state",
                "scored_observations": "window endpoints from window_size - 1 onward, all detectors",
            },
            indent=2,
        )
        + "\n"
    )

    def checkpoint(new_rows: List[Dict[str, object]]) -> None:
        rows.extend(new_rows)
        pd.DataFrame(rows).to_csv(replications_path, index=False)
        first = new_rows[0]
        summary = {(r["detector"], r["learner"]): r for r in new_rows if r["target"] == 1.0}
        print(
            "{} replication {:2d} ({:.0f}s): FA=1/1000 DP  CUSUM {:.2f}  HMM {:.2f}  Stat/RF {:.2f}  Stat+raw/logit {:.2f}  B3+raw/logit {:.2f}".format(
                design,
                first["replication"],
                first["elapsed_seconds"],
                summary[("cusum", "none")]["test_detection_probability"],
                summary[("hmm_filter", "none")]["test_detection_probability"],
                summary[("statistics", "random_forest")]["test_detection_probability"],
                summary[("statistics_rawsig", "logistic")]["test_detection_probability"],
                summary[("b3_rawsig", "logistic")]["test_detection_probability"],
            ),
            flush=True,
        )

    if jobs == 1:
        for task in tasks:
            checkpoint(evaluate_replication(task))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(evaluate_replication, task) for task in tasks]
            for future in as_completed(futures):
                checkpoint(future.result())

    results = attach_selected_learner(pd.DataFrame(rows), grid_results)
    results = results.sort_values(["replication", "detector", "learner", "target"]).reset_index(drop=True)
    results.to_csv(replications_path, index=False)
    summary = summarize(results)
    summary.to_csv(output_dir / "summary.csv", index=False)
    paired_contrasts(results).to_csv(output_dir / "contrasts.csv", index=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--designs", nargs="+", choices=tuple(grid.DESIGNS), default=list(grid.DESIGNS))
    parser.add_argument("--grid-root", type=Path, default=Path("results/unified_grid"))
    parser.add_argument("--output-root", type=Path, default=Path("results/unified_grid/detector_matched_fa"))
    parser.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--figure-only", action="store_true", help="Rebuild the figure from saved summaries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    summaries = {}
    for design in args.designs:
        if args.figure_only:
            summaries[design] = pd.read_csv(args.output_root / design / "summary.csv")
        else:
            summaries[design] = run_design(
                design, args.grid_root.resolve(), args.output_root.resolve(), args.jobs, args.tolerance, args.resume
            )
    make_figure(summaries, args.output_root / "figure_tradeoff.png", args.tolerance)
    for design, summary in summaries.items():
        print("\n{}: detection probability at each target (selected learner / baselines)".format(design))
        view = summary[summary["learner"].isin(["selected", "none"])].pivot_table(
            index="detector", columns="target", values="detection_probability"
        )
        print(view.round(3).to_string())
    print("\nTotal time: {:.1f}s".format(time.monotonic() - started))


if __name__ == "__main__":
    main()
