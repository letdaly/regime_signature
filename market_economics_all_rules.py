"""Volatility-managed results for every signature rule of the market study.

The market study reports the two signature rules fixed before the market data were
analyzed (Statistics plus log-signatures and B3 plus raw signatures) together with the
Statistics and HMM benchmarks. This script adds the remaining two rules, Statistics plus
raw signatures and B3 plus log-signatures, from the out-of-sample weights stored in
``results/market_french/<market>/folds``; no model is refitted.

The published rules keep their position in the rule list, so their Sharpe-ratio tests
reuse the same bootstrap seeds, and the script checks that every published row is
reproduced exactly before writing anything.

    python market_economics_all_rules.py --output-dir reproduced/market_economics_all_rules
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import experiment_market_french as market

EXTRA_CELLS = ("statistics_rawsig", "b3_logsig")
MARKETS = ("us", "developed_ex_us")


def economics_for_market(results_root: Path, name: str) -> pd.DataFrame:
    market_dir = results_root / name
    config = json.loads((market_dir / "config.json").read_text())
    cells = list(config["cells"])
    pooled = market.load_folds(market_dir / "folds", cells)
    rules = [cell for cell in market.ECONOMIC_CELLS if cell in cells] + list(EXTRA_CELLS)
    table = market.economics_table(
        pooled,
        rules,
        market.SHARPE_BENCHMARK_CELL,
        int(config["bootstrap_block"]),
        int(config["sharpe_resamples"]),
        int(config["seed"]),
    )
    table.insert(0, "market", name)

    published = pd.read_csv(market_dir / "economics.csv")
    merged = published.merge(table, on=["market", "rule"], suffixes=("_published", "_new"))
    if len(merged) != len(published):
        raise RuntimeError("{}: a published rule is missing from the recomputation.".format(name))
    for column in published.columns.drop(["market", "rule"]):
        old = merged[column + "_published"].to_numpy(dtype=float)
        new = merged[column + "_new"].to_numpy(dtype=float)
        if not np.allclose(old, new, rtol=1e-12, atol=1e-12, equal_nan=True):
            raise RuntimeError("{}: column {} differs from economics.csv.".format(name, column))
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", type=Path, default=Path("results/market_french"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/market_economics_all_rules"))
    args = parser.parse_args()

    table = pd.concat([economics_for_market(args.results_root, name) for name in MARKETS], ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "economics.csv", index=False)
    configuration = {
        "experiment": "market_economics_all_rules",
        "source": str(args.results_root),
        "markets": list(MARKETS),
        "published_rules": list(market.ECONOMIC_CELLS),
        "added_rules": list(EXTRA_CELLS),
        "sharpe_benchmark": market.SHARPE_BENCHMARK_CELL,
        "published_rows_reproduced": True,
    }
    (args.output_dir / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
