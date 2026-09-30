"""Volatility-managed exposure driven by the HAR forecasts, with and without signatures.

Applies the exposure rule of the market study, w_t = c / sigma_hat_t^2 capped at two, to the
tercile probabilities of the plain and augmented HAR regressions of ``market_har_signatures.py``.
Each regression is refitted exactly as there, with the penalty recorded in
``results/market_har_signatures/penalties.csv``; its in-sample probabilities on the training
window set the scale c, as the classifiers' do in the market study, and its test probabilities
must reproduce the stored forecasts. Sharpe ratios are compared with the plain HAR rule by the
studentized bootstrap test of the market study, with Holm correction across the six comparisons
of the two markets. This evaluation was added after the augmented forecasts had been scored; the
exposure rule itself is the one fixed before the market analysis.

    python market_har_economics.py --output-dir reproduced/market_har_economics
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import experiment_market_french as market
from market_forecast_evaluation import MARKETS, holm
from market_har_signatures import EXTRA_BLOCKS, HAR_NAMES, extra_columns, har_design, rebuild_full, ridge_forecast

MODELS = ("har",) + tuple("har_" + block for block in EXTRA_BLOCKS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", type=Path, default=Path("results/market_french"))
    parser.add_argument("--har-root", type=Path, default=Path("results/market_har_signatures"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/french"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/market_har_economics"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    penalties = pd.read_csv(args.har_root / "penalties.csv")
    penalty_of = {(row.market, int(row.year), row.model): float(row.penalty) for row in penalties.itertuples()}
    tables = []
    for name in MARKETS:
        config, data, folds = rebuild_full(name, args.results_root, args.data_dir)
        columns = extra_columns(data.feature_names)
        har = np.asarray(data.X[:, [data.feature_names.index(n) for n in HAR_NAMES]], dtype=float)
        target = np.log(data.forward_rv)
        probabilities = {model: [] for model in MODELS}
        weights = {model: [] for model in MODELS}
        excess = []
        for fold in folds:
            design = har_design(har, fold.pool)
            for model in MODELS:
                if model == "har":
                    extra, penalty = np.empty((len(target), 0)), np.inf
                else:
                    extra = np.asarray(data.X[:, columns[model[len("har_"):]]], dtype=float)
                    penalty = penalty_of[(name, fold.year, model)]
                test = ridge_forecast(design, extra, target, fold.pool, fold.test, penalty, fold.cut_points)
                pool = ridge_forecast(design, extra, target, fold.pool, fold.pool, penalty, fold.cut_points)
                probabilities[model].append(test)
                weights[model].append(market.exposure_weights(data, fold, pool, test))
            excess.append(data.excess_returns[data.end_index[fold.test] + 1])

        stored = np.load(args.har_root / "forecasts_{}.npz".format(name))
        for model in MODELS:
            if not np.allclose(np.concatenate(probabilities[model]), stored["prob_" + model], rtol=0, atol=1e-12):
                raise RuntimeError("{}: {} does not reproduce the stored forecasts.".format(name, model))
        classifier = market.load_folds(args.results_root / name / "folds", ["statistics"])
        excess = np.concatenate(excess)
        if not np.array_equal(excess, classifier["excess_next"]):
            raise RuntimeError("{}: next-day excess returns differ from the market study.".format(name))

        pooled = {"excess_next": excess, "weights": {model: np.concatenate(weights[model]) for model in MODELS}}
        seed = int(config["seed"]) + 5000
        table = market.economics_table(
            pooled, MODELS, "har", int(config["bootstrap_block"]), int(config["sharpe_resamples"]), seed
        )
        # Context only, outside the Holm family: the plain HAR rule against the Statistics rule.
        context = market.sharpe_test(
            excess * pooled["weights"]["har"],
            excess * classifier["weights"]["statistics"],
            int(config["bootstrap_block"]),
            int(config["sharpe_resamples"]),
            seed + 100,
        )
        table["p_har_vs_statistics_rule"] = np.where(table["rule"] == "har", context["p_value"], np.nan)
        table.insert(0, "market", name)
        tables.append(table)

    economics = pd.concat(tables, ignore_index=True)
    compared = economics["rule"].str.startswith("har_")
    economics["p_holm"] = np.nan
    economics.loc[compared, "p_holm"] = holm(economics.loc[compared, "p_value"]).to_numpy()
    economics.to_csv(args.output_dir / "economics.csv", index=False)
    (args.output_dir / "config.json").write_text(json.dumps({
        "experiment": "market_har_economics",
        "forecasts": str(args.har_root),
        "rules": list(MODELS),
        "sharpe_benchmark": "har",
        "exposure_rule": "w_t = c / sigma_hat_t^2, capped at {}".format(market.LEVERAGE_CAP),
        "scale_from": "in-sample probabilities on the training window",
        "holm_family": "six HAR augmentations across the two markets",
        "stored_forecasts_reproduced": True,
    }, indent=2) + "\n")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(economics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
