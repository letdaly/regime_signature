"""Causal regime-transition detector and event-level metrics.

Shared by the Experiment A and Experiment B runners (and any future fBM-driven
row) so that every cell of the main table reports the same state metrics,
detection delay, and false-switch rate.  Nothing here looks at test data
during tuning: ``tune_detector`` is applied to validation windows only and the
selected setting is then evaluated once on the test windows.
"""

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

from regime_pipeline import SimulatedPath, WindowDataset


REGIME_IDS = (0, 1, 2)

# Paired contrasts reported for every representation against Statistics.
DETECTOR_CONTRAST_METRICS = (
    "detector_balanced_accuracy",
    "boundary_f1",
    "detection_probability",
    "false_switches_per_1000",
    "mean_detection_delay",
)

# Windows are stratified by the number of observations since the most recent
# true regime change at the window endpoint.  With a 50-step causal window:
# 0-24 means more than half of the window still belongs to the previous regime,
# 25-49 less than half, 50-99 the window lies entirely in the new regime but is
# recent, 100+ is mature.
TIME_SINCE_CHANGE_BUCKETS: Tuple[Tuple[str, int, float], ...] = (
    ("0_24", 0, 24),
    ("25_49", 25, 49),
    ("50_99", 50, 99),
    ("100_plus", 100, np.inf),
)


def time_since_last_change(
    dataset: WindowDataset, paths: Iterable[SimulatedPath]
) -> np.ndarray:
    """Observations between each window endpoint and the latest true change.

    A change is the first observation of a new regime, so a window ending on
    that observation has distance zero.  Windows that end before any change
    in their path are assigned ``inf``: their whole history lies in one
    regime, which is the mature case rather than a recent transition.
    """

    frames = {path.path_id: path.frame["Regime"].to_numpy() for path in paths}
    distances = np.empty(len(dataset), dtype=float)
    for path_id in dict.fromkeys(dataset.path_ids.tolist()):
        if path_id not in frames:
            raise KeyError("No simulated path for window provenance {}.".format(path_id))
        regimes = frames[path_id]
        change_times = np.flatnonzero(regimes[1:] != regimes[:-1]) + 1
        mask = dataset.path_ids == path_id
        end_times = dataset.end_times[mask]
        if len(change_times) == 0:
            # A path that never switches: every window is mature.
            distances[mask] = np.inf
            continue
        # Index of the latest change at or before each endpoint.
        latest = np.searchsorted(change_times, end_times, side="right") - 1
        distance = np.where(
            latest >= 0,
            end_times - change_times[np.maximum(latest, 0)],
            np.inf,
        )
        distances[mask] = distance
    return distances


def bucket_of_time_since_change(distances: np.ndarray) -> np.ndarray:
    """Map distances to bucket names from TIME_SINCE_CHANGE_BUCKETS."""

    labels = np.empty(len(distances), dtype=object)
    for name, low, high in TIME_SINCE_CHANGE_BUCKETS:
        labels[(distances >= low) & (distances <= high)] = name
    if any(label is None for label in labels):
        raise ValueError("Every distance must fall into one bucket.")
    return labels


def stratified_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    distances: np.ndarray,
    prefix: str,
) -> Dict[str, float]:
    """Balanced accuracy, accuracy, and window counts per time-since-change bucket.

    Balanced accuracy averages recall over the regimes present in the bucket;
    buckets without windows report NaN and a zero count.
    """

    buckets = bucket_of_time_since_change(distances)
    diagnostics: Dict[str, float] = {}
    for name, _, _ in TIME_SINCE_CHANGE_BUCKETS:
        mask = buckets == name
        count = int(np.sum(mask))
        diagnostics[prefix + "_n_since_" + name] = count
        if count == 0:
            diagnostics[prefix + "_ba_since_" + name] = np.nan
            diagnostics[prefix + "_accuracy_since_" + name] = np.nan
            continue
        # Balanced accuracy = mean recall over the regimes present in the
        # bucket; computed explicitly so single-regime buckets do not warn.
        present = np.unique(y_true[mask])
        diagnostics[prefix + "_ba_since_" + name] = float(
            recall_score(
                y_true[mask], y_pred[mask], labels=present, average="macro", zero_division=0
            )
        )
        diagnostics[prefix + "_accuracy_since_" + name] = float(
            accuracy_score(y_true[mask], y_pred[mask])
        )
    return diagnostics


DETECTOR_GRID = tuple(
    (alpha, threshold, confirmations)
    for alpha in (0.2, 0.5, 1.0)
    for threshold in (0.40, 0.50, 0.60)
    for confirmations in (1, 3, 5)
)


def ordered_probabilities(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """Return probabilities in fixed regime order, even if a class is absent."""

    probabilities = np.zeros((len(X), len(REGIME_IDS)), dtype=float)
    fitted = model.predict_proba(X)
    for source_column, regime in enumerate(model.classes_.astype(int)):
        probabilities[:, regime] = fitted[:, source_column]
    return probabilities




def causal_detector(
    probabilities: np.ndarray,
    alpha: float,
    threshold: float,
    confirmations: int,
) -> np.ndarray:
    """Convert probabilities to a causal, persistent sequence of state decisions."""

    if len(probabilities) == 0:
        return np.empty(0, dtype=int)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must lie in (0, 1].")
    if not 0.0 <= threshold <= 1.0 or confirmations < 1:
        raise ValueError("Invalid detector threshold or confirmation count.")

    smoothed = probabilities[0].astype(float, copy=True)
    current = int(np.argmax(smoothed))
    candidate = current
    candidate_count = 0
    decisions = np.empty(len(probabilities), dtype=int)
    decisions[0] = current

    for index in range(1, len(probabilities)):
        smoothed = alpha * probabilities[index] + (1.0 - alpha) * smoothed
        proposed = int(np.argmax(smoothed))
        confident = float(smoothed[proposed]) >= threshold
        if proposed != current and confident:
            if proposed == candidate:
                candidate_count += 1
            else:
                candidate = proposed
                candidate_count = 1
            if candidate_count >= confirmations:
                current = candidate
                candidate_count = 0
        else:
            candidate = current
            candidate_count = 0
        decisions[index] = current
    return decisions


def _change_events(
    states: np.ndarray, end_times: np.ndarray
) -> List[Tuple[int, int]]:
    indices = np.flatnonzero(states[1:] != states[:-1]) + 1
    return [(int(end_times[index]), int(states[index])) for index in indices]


def _match_events(
    true_events: Sequence[Tuple[int, int]],
    predicted_events: Sequence[Tuple[int, int]],
    tolerance: int,
) -> Tuple[int, List[int], int]:
    """Greedily match causal predictions to same-state changes within tolerance."""

    used = set()
    delays: List[int] = []
    for true_time, new_state in true_events:
        for predicted_index, (predicted_time, predicted_state) in enumerate(predicted_events):
            if predicted_index in used:
                continue
            if predicted_time < true_time:
                continue
            if predicted_time > true_time + tolerance:
                break
            if predicted_state == new_state:
                used.add(predicted_index)
                delays.append(predicted_time - true_time)
                break
    return len(delays), delays, len(predicted_events) - len(used)


def detector_metrics(
    dataset: WindowDataset,
    probabilities: np.ndarray,
    alpha: float,
    threshold: float,
    confirmations: int,
    tolerance: int,
) -> Dict[str, float]:
    """Aggregate causal state, calibration, and transition metrics by path."""

    all_truth = []
    all_decisions = []
    matched_events = 0
    total_true_events = 0
    false_events = 0
    delays: List[int] = []

    for path_id in dict.fromkeys(dataset.path_ids.tolist()):
        mask = dataset.path_ids == path_id
        truth = dataset.y[mask]
        times = dataset.end_times[mask]
        path_probabilities = probabilities[mask]
        decisions = causal_detector(
            path_probabilities, alpha, threshold, confirmations
        )
        true_events = _change_events(truth, times)
        predicted_events = _change_events(decisions, times)
        matched, path_delays, false = _match_events(
            true_events, predicted_events, tolerance
        )
        matched_events += matched
        total_true_events += len(true_events)
        false_events += false
        delays.extend(path_delays)
        all_truth.append(truth)
        all_decisions.append(decisions)

    truth = np.concatenate(all_truth)
    decisions = np.concatenate(all_decisions)
    precision_denominator = matched_events + false_events
    precision = matched_events / precision_denominator if precision_denominator else 0.0
    recall = matched_events / total_true_events if total_true_events else 0.0
    boundary_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    targets = np.eye(len(REGIME_IDS), dtype=float)[dataset.y]
    brier = float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))
    return {
        "detector_balanced_accuracy": float(balanced_accuracy_score(truth, decisions)),
        "detector_macro_f1": float(
            f1_score(truth, decisions, labels=REGIME_IDS, average="macro", zero_division=0)
        ),
        "fraction_time_wrong": float(np.mean(truth != decisions)),
        "brier_score": brier,
        "detection_probability": float(recall),
        "mean_detection_delay": float(np.mean(delays)) if delays else np.nan,
        "median_detection_delay": float(np.median(delays)) if delays else np.nan,
        "false_switches_per_1000": float(false_events * 1000.0 / len(truth)),
        "boundary_precision": float(precision),
        "boundary_recall": float(recall),
        "boundary_f1": float(boundary_f1),
        "n_true_changes": int(total_true_events),
        "n_matched_changes": int(matched_events),
        "n_false_switches": int(false_events),
    }


def tune_detector(
    dataset: WindowDataset,
    probabilities: np.ndarray,
    tolerance: int,
) -> Tuple[Tuple[float, float, int], Dict[str, float]]:
    """Select a detector on validation data using event F1, then delay/alarms."""

    best_parameters = DETECTOR_GRID[0]
    best_metrics: Dict[str, float] = {}
    best_score = None
    for alpha, threshold, confirmations in DETECTOR_GRID:
        metrics = detector_metrics(
            dataset,
            probabilities,
            alpha=alpha,
            threshold=threshold,
            confirmations=confirmations,
            tolerance=tolerance,
        )
        delay = metrics["mean_detection_delay"]
        score = (
            metrics["boundary_f1"],
            metrics["detection_probability"],
            -metrics["false_switches_per_1000"],
            -delay if np.isfinite(delay) else -np.inf,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_parameters = (alpha, threshold, confirmations)
            best_metrics = metrics
    return best_parameters, best_metrics


