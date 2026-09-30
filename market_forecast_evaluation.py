"""Probabilistic evaluation of the market forecasts against conventional benchmarks.

For each market the script rebuilds the walk-forward folds of ``experiment_market_french``
(same frozen data, windows, tercile cut-points and embargo) without recomputing any signature,
and checks that every test date and label matches the stored fold files exactly. It then fits
four benchmark forecasts on each fold's training window (training and validation years, as for
the refitted classifiers):

* climatology: the training frequencies of the three terciles;
* persistence: the tercile of realized volatility over the last 21 days, as a one-hot forecast;
* transition: the training frequencies of next-month terciles given the current tercile
  (Laplace-smoothed);
* HAR: a regression of log next-month realized volatility on the logs of realized volatility
  over the last 1, 5 and 22 days, with Gaussian residuals turning the forecast into tercile
  probabilities. Days with a zero return would give a log of zero, so the one-day component is
  floored at the 1% quantile of its training values.

The benchmarks and the seven stored representations are scored by balanced accuracy, the ranked
probability score (RPS, the primary score for ordered terciles), the logarithmic score and the
Brier score. Signature augmentations are compared with their base block, and every forecast
with the HAR benchmark, by Diebold-Mariano tests (Bartlett HAC variance, bandwidth 21) and
stationary-bootstrap intervals (mean block length 21), with Holm correction within each family.

    python market_forecast_evaluation.py --output-dir reproduced/market_forecast_evaluation
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import norm

import experiment_market_french as market

MARKETS = ("us", "developed_ex_us")
BENCHMARKS = ("climatology", "persistence", "transition", "har")
STORED = ("hmm", "statistics", "statistics_logsig", "statistics_rawsig", "b3", "b3_logsig", "b3_rawsig")
AUGMENTATIONS = (
    ("statistics_logsig", "statistics"),
    ("statistics_rawsig", "statistics"),
    ("b3_logsig", "b3"),
    ("b3_rawsig", "b3"),
)
SCORES = ("rps", "log_score", "brier")
TERCILES = market.N_TERCILES


def rebuild(name: str, results_root: Path, data_dir: Path):
    config = json.loads((results_root / name / "config.json").read_text())
    frame, provenance = market.load_market(name, data_dir, config["sample_start"], config["sample_end"], False)
    if provenance["sha256"] != config["data"]["sha256"]:
        raise RuntimeError("{}: the local data file differs from the frozen snapshot.".format(name))
    returns = frame["LogReturn"].to_numpy(dtype=float)
    end_index = np.arange(int(config["window_size"]) - 1, len(returns))
    har, har_names = market.har_features(returns, end_index)
    data = market.MarketDataset(
        market=name,
        dates=pd.DatetimeIndex(frame.index),
        log_returns=returns,
        excess_returns=frame["ExcessReturn"].to_numpy(dtype=float),
        X=har,
        feature_names=har_names,
        end_index=end_index,
        forward_rv=market.forward_realized_volatility(returns, end_index, int(config["horizon"])),
    )
    folds = market.build_folds(
        data,
        int(config["first_test_year"]),
        int(config["last_test_year"]),
        int(config["horizon"]),
        int(config["validation_years"]),
        int(config["minimum_train_windows"]),
    )
    return config, data, folds


def trailing_volatility(returns: np.ndarray, end_index: np.ndarray, horizon: int) -> np.ndarray:
    """Annualized realized volatility over the ``horizon`` days ending at each endpoint."""

    cumulative = np.concatenate(([0.0], np.cumsum(returns ** 2)))
    values = np.full(len(end_index), np.nan)
    complete = end_index + 1 >= horizon
    ends = end_index[complete] + 1
    values[complete] = np.sqrt(market.TRADING_DAYS * (cumulative[ends] - cumulative[ends - horizon]) / horizon)
    return values


def benchmark_probabilities(config, data, fold) -> Dict[str, np.ndarray]:
    pool = fold.pool
    test = fold.test
    labels = fold.labels
    horizon = int(config["horizon"])

    counts = np.bincount(labels[pool], minlength=TERCILES).astype(float)
    climatology = np.tile(counts / counts.sum(), (int(test.sum()), 1))

    current = np.digitize(trailing_volatility(data.log_returns, data.end_index, horizon), fold.cut_points)
    persistence = np.eye(TERCILES)[current[test]]

    transitions = np.ones((TERCILES, TERCILES))
    np.add.at(transitions, (current[pool], labels[pool]), 1.0)
    transition = (transitions / transitions.sum(axis=1, keepdims=True))[current[test]]

    har = np.asarray(data.X, dtype=float).copy()
    floor = np.quantile(har[pool, 0], 0.01)
    har[:, 0] = np.maximum(har[:, 0], floor)
    design = np.column_stack((np.ones(len(har)), 0.5 * np.log(market.TRADING_DAYS * har)))
    if not np.isfinite(design[pool | test]).all():
        raise FloatingPointError("Non-finite HAR regressor in {}.".format(fold.year))
    target = np.log(data.forward_rv)
    coefficients, *_ = np.linalg.lstsq(design[pool], target[pool], rcond=None)
    # The regressors are finite (checked above); the guard only silences the spurious
    # Accelerate matmul warnings noted in experiment_unified_grid.fit_cell.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        residuals = target[pool] - design[pool] @ coefficients
        mean = design[test] @ coefficients
    scale = float(np.std(residuals, ddof=design.shape[1]))
    below = norm.cdf((np.log(fold.cut_points)[None, :] - mean[:, None]) / scale)
    har_probabilities = np.column_stack((below[:, 0], below[:, 1] - below[:, 0], 1.0 - below[:, 1]))
    if not (np.isfinite(scale) and np.isfinite(har_probabilities).all()
            and np.allclose(har_probabilities.sum(axis=1), 1.0)):
        raise FloatingPointError("HAR forecast is not a valid distribution in {}.".format(fold.year))

    return {
        "climatology": climatology,
        "persistence": persistence,
        "transition": transition,
        "har": har_probabilities,
    }


def daily_scores(probabilities: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
    outcome = np.eye(TERCILES)[labels]
    cumulative_gap = np.cumsum(probabilities, axis=1)[:, :-1] - np.cumsum(outcome, axis=1)[:, :-1]
    realized = probabilities[np.arange(len(labels)), labels]
    with np.errstate(divide="ignore"):
        log_score = -np.log(realized)
    return {
        "rps": np.sum(cumulative_gap ** 2, axis=1) / (TERCILES - 1),
        "log_score": log_score,
        "brier": np.sum((probabilities - outcome) ** 2, axis=1),
    }


def compare(loss_base: np.ndarray, loss_other: np.ndarray, block: float, resamples: int, seed: int) -> Dict[str, float]:
    """Mean loss reduction of ``other`` relative to ``base``, with DM and bootstrap inference."""

    difference = np.asarray(loss_base - loss_other, dtype=float)
    if not np.all(np.isfinite(difference)):
        return {"improvement": float("nan")}
    point = float(difference.mean())
    variance = float(market._hac_covariance(difference[:, None], block)[0, 0]) / len(difference)
    statistic = point / np.sqrt(variance) if variance > 0 else float("nan")
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [difference[index].mean() for index in market.stationary_bootstrap_indices(len(difference), block, resamples, rng)]
    )
    low, high = np.quantile(draws, (0.025, 0.975))
    return {
        "improvement": point,
        "ci_low": 2.0 * point - high,
        "ci_high": 2.0 * point - low,
        "dm_statistic": float(statistic),
        "p_dm": float(2.0 * (1.0 - norm.cdf(abs(statistic)))) if np.isfinite(statistic) else float("nan"),
    }


def holm(p_values: pd.Series) -> pd.Series:
    order = np.argsort(p_values.to_numpy())
    adjusted = np.empty(len(order))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * float(p_values.iloc[index])))
        adjusted[index] = running
    return pd.Series(adjusted, index=p_values.index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", type=Path, default=Path("results/market_french"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/french"))
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=Path("results/market_forecast_evaluation"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    levels, contrasts = [], []
    for name in MARKETS:
        config, data, folds = rebuild(name, args.results_root, args.data_dir)
        stored = market.load_folds(args.results_root / name / "folds", list(STORED))
        parts = {benchmark: [] for benchmark in BENCHMARKS}
        dates, labels = [], []
        for fold in folds:
            dates.append(data.window_dates[fold.test].asi8)
            labels.append(fold.labels[fold.test])
            for benchmark, probabilities in benchmark_probabilities(config, data, fold).items():
                parts[benchmark].append(probabilities)
        dates, labels = np.concatenate(dates), np.concatenate(labels)
        if not (np.array_equal(dates, stored["dates"].asi8) and np.array_equal(labels, stored["labels"])):
            raise RuntimeError("{}: rebuilt test dates or labels differ from the stored folds.".format(name))
        forecasts = {benchmark: np.concatenate(values) for benchmark, values in parts.items()}
        forecasts.update({cell: np.asarray(stored["probabilities"][cell], dtype=float) for cell in STORED})
        np.savez_compressed(
            args.output_dir / "benchmarks_{}.npz".format(name),
            dates=dates, labels=labels, **{"prob_" + b: forecasts[b] for b in BENCHMARKS}
        )

        scores = {model: daily_scores(probabilities, labels) for model, probabilities in forecasts.items()}
        reference = scores["climatology"]["rps"].mean()
        for model, probabilities in forecasts.items():
            ba = market.balanced_accuracy(labels, np.argmax(probabilities, axis=1))
            row = {"market": name, "model": model, "n": int(len(labels)),
                   "ba_pct": float("nan") if model == "climatology" else 100.0 * ba}
            for score in SCORES:
                row[score] = float(scores[model][score].mean())
            row["rps_skill_pct"] = 100.0 * (1.0 - row["rps"] / reference)
            levels.append(row)

        seed = int(config["seed"])
        for offset, (augmented, base) in enumerate(AUGMENTATIONS):
            for s_index, score in enumerate(SCORES):
                row = {"market": name, "family": "augmentation", "model": augmented, "reference": base, "score": score}
                row.update(compare(scores[base][score], scores[augmented][score], market.BOOTSTRAP_BLOCK,
                                   args.resamples, seed + 10 * offset + s_index))
                contrasts.append(row)
        predictions = {model: np.argmax(probabilities, axis=1) for model, probabilities in forecasts.items()}
        for offset, model in enumerate(("persistence",) + STORED):
            gain = market.bootstrap_gain(labels, predictions[model], predictions["har"], market.BOOTSTRAP_BLOCK,
                                         args.resamples, seed + 2000 + offset)
            contrasts.append({"market": name, "family": "versus_har", "model": model, "reference": "har",
                              "score": "balanced_accuracy", "improvement": gain["gain_points"],
                              "ci_low": gain["ci_low"], "ci_high": gain["ci_high"], "p_value": gain["bootstrap_p"]})
        for offset, model in enumerate(("transition",) + STORED):
            for s_index, score in enumerate(SCORES):
                row = {"market": name, "family": "versus_har", "model": model, "reference": "har", "score": score}
                row.update(compare(scores["har"][score], scores[model][score], market.BOOTSTRAP_BLOCK,
                                   args.resamples, seed + 1000 + 10 * offset + s_index))
                contrasts.append(row)

    levels = pd.DataFrame(levels)
    contrasts = pd.DataFrame(contrasts)
    contrasts["p_value"] = contrasts["p_value"].fillna(contrasts["p_dm"])
    contrasts["p_holm"] = np.nan
    for (family, score), group in contrasts.groupby(["family", "score"]):
        valid = group["p_value"].notna()
        contrasts.loc[group.index[valid], "p_holm"] = holm(group.loc[valid, "p_value"]).to_numpy()
    levels.to_csv(args.output_dir / "levels.csv", index=False)
    contrasts.to_csv(args.output_dir / "contrasts.csv", index=False)
    (args.output_dir / "config.json").write_text(json.dumps({
        "experiment": "market_forecast_evaluation",
        "source": str(args.results_root),
        "markets": list(MARKETS),
        "benchmarks": list(BENCHMARKS),
        "stored_representations": list(STORED),
        "scores": list(SCORES),
        "primary_score": "rps",
        "balanced_accuracy_versus_har": "stationary-bootstrap gain, as in the market study",
        "hac_bandwidth": market.BOOTSTRAP_BLOCK,
        "bootstrap_mean_block": market.BOOTSTRAP_BLOCK,
        "bootstrap_resamples": args.resamples,
        "har_one_day_floor_quantile": 0.01,
        "transition_smoothing": "Laplace (+1)",
        "rebuilt_dates_and_labels_match_stored_folds": True,
    }, indent=2) + "\n")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(levels.round(4).to_string(index=False))
        print(contrasts.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
