"""HAR regressions augmented with path signatures in the market forecasting study.

Tests whether signatures improve the strongest conventional benchmark of
``market_forecast_evaluation.py``, the log-HAR regression. For each market and test year the
script rebuilds the walk-forward folds of the market study with every feature block (the rebuilt
test dates and labels must match the stored folds) and fits

    log RV(t+1..t+21) = a + b_d log RV_d + b_w log RV_w + b_m log RV_m + c' z_t + e_t,

where z_t holds standardized extra predictors: the 14 level-3 log-signature coordinates, the 39
level-3 raw signature coordinates or, as a conventional control, the 11 Statistics features.
Only c is penalized (ridge). The penalty is chosen on the validation years by the ranked
probability score, from a grid that includes an infinite penalty, which returns the plain HAR
forecast; the model is then refitted on the full training window, and Gaussian residuals turn
the forecast into tercile probabilities, exactly as for the HAR benchmark. The design was fixed
before any augmented forecast was scored.

    python market_har_signatures.py --output-dir reproduced/market_har_signatures
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

import experiment_market_french as market
from experiment_unified_grid import STATISTICS_COLUMNS
from market_forecast_evaluation import MARKETS, SCORES, compare, daily_scores, holm

EXTRA_BLOCKS = ("statistics", "logsig", "rawsig")
PENALTIES = (0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, np.inf)
HAR_NAMES = tuple("b1_har_squared_return_{}".format(lag) for lag in market.HAR_LAGS)


def rebuild_full(name: str, results_root: Path, data_dir: Path):
    config = json.loads((results_root / name / "config.json").read_text())
    frame, provenance = market.load_market(name, data_dir, config["sample_start"], config["sample_end"], False)
    if provenance["sha256"] != config["data"]["sha256"]:
        raise RuntimeError("{}: the local data file differs from the frozen snapshot.".format(name))
    data = market.build_market_dataset(name, frame, int(config["window_size"]), int(config["horizon"]))
    folds = market.build_folds(
        data,
        int(config["first_test_year"]),
        int(config["last_test_year"]),
        int(config["horizon"]),
        int(config["validation_years"]),
        int(config["minimum_train_windows"]),
    )
    return config, data, folds


def extra_columns(feature_names) -> Dict[str, np.ndarray]:
    position = {name: index for index, name in enumerate(feature_names)}
    blocks = market.market_block_columns(feature_names)
    return {
        "statistics": np.asarray([position[name] for name in STATISTICS_COLUMNS], dtype=int),
        "logsig": blocks["logsig"],
        "rawsig": blocks["rawsig"],
    }


def har_design(har: np.ndarray, fit: np.ndarray) -> np.ndarray:
    """Intercept and log annualized 1-, 5- and 22-day volatility, one-day part floored on ``fit``."""

    values = har.copy()
    values[:, 0] = np.maximum(values[:, 0], np.quantile(values[fit, 0], 0.01))
    return np.column_stack((np.ones(len(values)), 0.5 * np.log(market.TRADING_DAYS * values)))


def ridge_forecast(
    design: np.ndarray, extra: np.ndarray, target: np.ndarray, fit: np.ndarray, predict: np.ndarray,
    penalty: float, cut_points: np.ndarray,
) -> np.ndarray:
    """Tercile probabilities of a HAR regression with ridge-penalized extra predictors."""

    if np.isinf(penalty):
        regressors_fit, regressors_predict = design[fit], design[predict]
        rows = regressors_fit
        response = target[fit]
    else:
        centre = extra[fit].mean(axis=0)
        spread = extra[fit].std(axis=0)
        spread[spread <= 0] = 1.0
        standardized = (extra - centre) / spread
        regressors_fit = np.column_stack((design[fit], standardized[fit]))
        regressors_predict = np.column_stack((design[predict], standardized[predict]))
        k = standardized.shape[1]
        penalty_rows = np.column_stack((np.zeros((k, design.shape[1])), np.sqrt(penalty) * np.eye(k)))
        rows = np.vstack((regressors_fit, penalty_rows))
        response = np.concatenate((target[fit], np.zeros(k)))
    coefficients, *_ = np.linalg.lstsq(rows, response, rcond=None)
    # Finite inputs; the guard only silences the spurious Accelerate matmul warnings.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        residuals = target[fit] - regressors_fit @ coefficients
        mean = regressors_predict @ coefficients
    scale = float(np.std(residuals, ddof=design.shape[1]))
    below = norm.cdf((np.log(cut_points)[None, :] - mean[:, None]) / scale)
    probabilities = np.column_stack((below[:, 0], below[:, 1] - below[:, 0], 1.0 - below[:, 1]))
    if not (np.isfinite(probabilities).all() and np.allclose(probabilities.sum(axis=1), 1.0)):
        raise FloatingPointError("Augmented HAR forecast is not a valid distribution.")
    return probabilities


def fold_forecasts(data, fold, columns) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    har = np.asarray(data.X[:, [data.feature_names.index(name) for name in HAR_NAMES]], dtype=float)
    target = np.log(data.forward_rv)
    forecasts, chosen = {}, {}
    forecasts["har"] = ridge_forecast(
        har_design(har, fold.pool), np.empty((len(target), 0)), target, fold.pool, fold.test, np.inf, fold.cut_points
    )
    for block in EXTRA_BLOCKS:
        extra = np.asarray(data.X[:, columns[block]], dtype=float)
        validation_rps = []
        for penalty in PENALTIES:
            probabilities = ridge_forecast(
                har_design(har, fold.train), extra, target, fold.train, fold.validation, penalty, fold.cut_points
            )
            validation_rps.append(daily_scores(probabilities, fold.labels[fold.validation])["rps"].mean())
        # Ties go to the larger penalty, i.e. towards the plain HAR forecast.
        best = min(range(len(PENALTIES)), key=lambda i: (round(validation_rps[i], 12), -i))
        chosen["har_" + block] = PENALTIES[best]
        forecasts["har_" + block] = ridge_forecast(
            har_design(har, fold.pool), extra, target, fold.pool, fold.test, PENALTIES[best], fold.cut_points
        )
    return forecasts, chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", type=Path, default=Path("results/market_french"))
    parser.add_argument("--benchmarks-root", type=Path, default=Path("results/market_forecast_evaluation"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/french"))
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=Path("results/market_har_signatures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    models = ("har",) + tuple("har_" + block for block in EXTRA_BLOCKS)
    levels, contrasts, selections = [], [], []
    for name in MARKETS:
        config, data, folds = rebuild_full(name, args.results_root, args.data_dir)
        stored = market.load_folds(args.results_root / name / "folds", ["statistics"])
        columns = extra_columns(data.feature_names)
        parts = {model: [] for model in models}
        dates, labels = [], []
        for fold in folds:
            dates.append(data.window_dates[fold.test].asi8)
            labels.append(fold.labels[fold.test])
            forecasts, chosen = fold_forecasts(data, fold, columns)
            for model in models:
                parts[model].append(forecasts[model])
            selections.extend({"market": name, "year": fold.year, "model": model, "penalty": penalty}
                              for model, penalty in chosen.items())
        dates, labels = np.concatenate(dates), np.concatenate(labels)
        if not (np.array_equal(dates, stored["dates"].asi8) and np.array_equal(labels, stored["labels"])):
            raise RuntimeError("{}: rebuilt test dates or labels differ from the stored folds.".format(name))
        forecasts = {model: np.concatenate(values) for model, values in parts.items()}
        benchmark = np.load(args.benchmarks_root / "benchmarks_{}.npz".format(name))
        if not np.allclose(forecasts["har"], benchmark["prob_har"], rtol=0, atol=1e-12):
            raise RuntimeError("{}: the plain HAR forecast differs from the HAR benchmark.".format(name))
        np.savez_compressed(args.output_dir / "forecasts_{}.npz".format(name), dates=dates, labels=labels,
                            **{"prob_" + model: forecasts[model] for model in models})

        scores = {model: daily_scores(probabilities, labels) for model, probabilities in forecasts.items()}
        for model, probabilities in forecasts.items():
            row = {"market": name, "model": model, "n": int(len(labels)),
                   "ba_pct": 100.0 * market.balanced_accuracy(labels, np.argmax(probabilities, axis=1))}
            row.update({score: float(scores[model][score].mean()) for score in SCORES})
            levels.append(row)

        seed = int(config["seed"]) + 3000
        predictions = {model: np.argmax(probabilities, axis=1) for model, probabilities in forecasts.items()}
        for offset, model in enumerate(models[1:]):
            gain = market.bootstrap_gain(labels, predictions[model], predictions["har"], market.BOOTSTRAP_BLOCK,
                                         args.resamples, seed + 100 + offset)
            contrasts.append({"market": name, "model": model, "reference": "har", "score": "balanced_accuracy",
                              "improvement": gain["gain_points"], "ci_low": gain["ci_low"],
                              "ci_high": gain["ci_high"], "p_value": gain["bootstrap_p"]})
            for s_index, score in enumerate(SCORES):
                row = {"market": name, "model": model, "reference": "har", "score": score}
                row.update(compare(scores["har"][score], scores[model][score], market.BOOTSTRAP_BLOCK,
                                   args.resamples, seed + 10 * offset + s_index))
                row["p_value"] = row.get("p_dm", float("nan"))
                contrasts.append(row)

    levels, contrasts, selections = pd.DataFrame(levels), pd.DataFrame(contrasts), pd.DataFrame(selections)
    contrasts["p_holm"] = np.nan
    for score, group in contrasts.groupby("score"):
        valid = group["p_value"].notna()
        contrasts.loc[group.index[valid], "p_holm"] = holm(group.loc[valid, "p_value"]).to_numpy()
    levels.to_csv(args.output_dir / "levels.csv", index=False)
    contrasts.to_csv(args.output_dir / "contrasts.csv", index=False)
    selections.to_csv(args.output_dir / "penalties.csv", index=False)
    (args.output_dir / "config.json").write_text(json.dumps({
        "experiment": "market_har_signatures",
        "source": str(args.results_root),
        "markets": list(MARKETS),
        "extra_blocks": {"statistics": len(STATISTICS_COLUMNS), "logsig": 14, "rawsig": 39},
        "penalties": [None if np.isinf(p) else p for p in PENALTIES],
        "penalty_selection": "validation-year mean RPS; ties to the larger penalty; None = plain HAR",
        "standardization": "extra predictors standardized on the fitting window",
        "primary_score": "rps",
        "hac_bandwidth": market.BOOTSTRAP_BLOCK,
        "bootstrap_mean_block": market.BOOTSTRAP_BLOCK,
        "bootstrap_resamples": args.resamples,
        "plain_har_matches_benchmark": True,
        "rebuilt_dates_and_labels_match_stored_folds": True,
    }, indent=2) + "\n")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(levels.round(4).to_string(index=False))
        print(contrasts.round(4).to_string(index=False))
        print(selections.groupby(["market", "model"])["penalty"].value_counts().unstack(fill_value=0))


if __name__ == "__main__":
    main()
