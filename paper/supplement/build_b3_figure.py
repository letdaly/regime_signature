"""Figure S1: year-by-year out-of-sample accuracy of B3 relative to Statistics.

Reads the stored walk-forward folds of the market study; nothing is refitted.
Run from the repository root:

    .venv/bin/python paper/supplement/build_b3_figure.py

Writes ``paper/supplement/figures/figure_b3_by_year.{pdf,png}`` and prints the
statistics quoted in the supplement.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
MARKETS = (("us", "United States"), ("developed_ex_us", "Developed ex US"))

SERIES = "#2a78d6"
BAND = "#cde2fb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"


def yearly(market):
    folds = pd.read_csv(ROOT / "results" / "market_french" / market / "folds.csv")
    table = folds[folds["selected"]].pivot(index="year", columns="cell", values="test_ba")
    return 100 * table


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    plt.rcParams.update({"font.size": 9, "axes.labelcolor": INK, "axes.titlecolor": INK})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharey="row")
    for column, (market, label) in enumerate(MARKETS):
        table = yearly(market)
        shortfall = table["b3"] - table["statistics"]
        gain = table["b3_rawsig"] - table["b3"]
        n = len(shortfall)
        mean = shortfall.mean()
        half = stats.t.ppf(0.975, n - 1) * shortfall.std(ddof=1) / np.sqrt(n)
        below = int((shortfall < 0).sum())
        r = np.corrcoef(shortfall, gain)[0, 1]
        print(
            "{}: B3 - Statistics mean {:+.2f} [{:+.2f}, {:+.2f}], sd {:.2f}, below in {}/{} years; "
            "sd Statistics {:.2f}, sd B3 {:.2f}; corr(B3+raw - B3, B3 - Statistics) {:.2f}".format(
                label, mean, mean - half, mean + half, shortfall.std(ddof=1), below, n,
                table["statistics"].std(ddof=1), table["b3"].std(ddof=1), r,
            )
        )

        ax = axes[0, column]
        style(ax)
        years = table.index.to_numpy()
        ax.axhspan(mean - half, mean + half, color=BAND, zorder=1, linewidth=0)
        ax.bar(years, shortfall, width=0.7, color=SERIES, zorder=2)
        ax.axhline(0, color=INK_SECONDARY, linewidth=0.8, zorder=3)
        ax.axhline(mean, color=INK, linewidth=1.2, linestyle="--", zorder=3)
        ax.set_title(label, fontsize=10, loc="left")
        ax.text(
            0.02, 0.95,
            "mean {:+.1f} [{:+.1f}, {:+.1f}]; below zero in {} of {} years".format(
                mean, mean - half, mean + half, below, n
            ),
            transform=ax.transAxes, fontsize=7.5, color=INK_SECONDARY,
        )
        ax.set_xticks(years[::4])
        ax.set_ylim(-20, 32)
        if column == 0:
            ax.set_ylabel("B3 minus Statistics (pp)")

        ax = axes[1, column]
        style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.axhline(0, color=INK_SECONDARY, linewidth=0.8)
        ax.axvline(0, color=INK_SECONDARY, linewidth=0.8)
        ax.scatter(shortfall, gain, s=28, color=SERIES, edgecolors="white", linewidths=0.8, zorder=3)
        ax.text(0.97, 0.94, "$r = {:.2f}$".format(r), transform=ax.transAxes, ha="right",
                fontsize=8, color=INK_SECONDARY)
        ax.set_xlabel("B3 minus Statistics (pp)")
        if column == 0:
            ax.set_ylabel("B3 + raw sig. minus B3 (pp)")

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / "figure_b3_by_year.{}".format(suffix), dpi=200)
    print("wrote", OUT / "figure_b3_by_year.pdf")


if __name__ == "__main__":
    main()
