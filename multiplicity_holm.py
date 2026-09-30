"""Holm-corrected multiplicity audit of the contrasts the paper reports.

Every interval in the paper is a nominal 95% interval for one contrast at a
time, and the paper reports many of them: 24 paired augmentation contrasts in
the simulation table, 32 in the detection study (four false-alarm budgets),
and 8 in the market accuracy table.  Reading "resolved" as a claim about a
whole table requires a correction for multiplicity, which matters most for the
detection contrasts, where a gain at one budget but not at another could arise
by chance.

This script recomputes each contrast from the frozen replication-level outputs
and applies the Holm step-down correction inside three families -- the three
tables -- and once more across all 64 contrasts as a stress test.  Nothing is
refitted and no result directory is rewritten; the output is a new table for
the supplement.

Families:

* ``classification``: two designs x three learners x four augmentations, paired
  t-tests on the 50 within-replication balanced-accuracy differences, the same
  differences that produce the Student-t intervals of the simulation table.
* ``detection``: two designs x four false-alarm budgets x four augmentations,
  paired t-tests on the within-replication detection-probability differences of
  the validation-selected learner.  Conditional-delay contrasts are left out
  because none is resolved even uncorrected, so they can only dilute the
  correction.
* ``market``: two markets x four augmentations, using the stationary-bootstrap
  p-values already stored with the walk-forward study.

Usage: ``python multiplicity_holm.py`` writes ``results/multiplicity/holm.csv``.
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

DESIGNS = (("scale_only", "Design A"), ("matched_marginal", "Design B"))
MARKETS = (("us", "United States"), ("developed_ex_us", "Developed ex US"))
LEARNERS = ("logistic", "random_forest", "gbm")
# The four augmentation contrasts the paper reports for every table.
AUGMENTATIONS = (
    ("statistics_logsig", "statistics"),
    ("statistics_rawsig", "statistics"),
    ("b3_logsig", "b3"),
    ("b3_rawsig", "b3"),
)
BUDGETS = (0.5, 1.0, 2.0, 5.0)
GRID_DIR = Path("results/unified_grid")
MARKET_DIR = Path("results/market_french")
ALPHA = 0.05


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values, monotone and capped at one."""

    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    n = len(values)
    running = 0.0
    adjusted = np.empty(n, dtype=float)
    for rank, index in enumerate(order):
        running = max(running, (n - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def paired_t(differences: Sequence[float]) -> Dict[str, float]:
    """Mean, 95% interval and two-sided p-value of a paired difference."""

    values = np.asarray(differences, dtype=float)
    n = len(values)
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / np.sqrt(n))
    critical = float(student_t.ppf(0.975, df=n - 1))
    if standard_error > 0:
        statistic = mean / standard_error
        p_value = float(2.0 * student_t.sf(abs(statistic), df=n - 1))
    else:
        # A difference with no variation across replications means the two cells
        # are the same column, not infinite evidence; leave it unresolved.
        p_value = 1.0
    return {
        "n": n,
        "delta_points": 100.0 * mean,
        "ci_low": 100.0 * (mean - critical * standard_error),
        "ci_high": 100.0 * (mean + critical * standard_error),
        "p_raw": p_value,
        "method": "paired t",
    }


def classification_contrasts() -> List[Dict[str, object]]:
    rows = []
    for design, label in DESIGNS:
        frame = pd.read_csv(GRID_DIR / design / "replications.csv").sort_values("replication")
        for learner in LEARNERS:
            for augmented, base in AUGMENTATIONS:
                differences = (
                    frame["{}_{}_test_ba".format(learner, augmented)]
                    - frame["{}_{}_test_ba".format(learner, base)]
                )
                row = {
                    "family": "classification",
                    "unit": label,
                    "learner": learner,
                    "contrast": "{} - {}".format(augmented, base),
                    "budget": "",
                }
                row.update(paired_t(differences))
                rows.append(row)
    return rows


def detection_contrasts() -> List[Dict[str, object]]:
    rows = []
    for design, label in DESIGNS:
        frame = pd.read_csv(
            GRID_DIR / "detector_matched_fa" / design / "replications.csv"
        )
        selected = frame[frame["learner"] == "selected"]
        for budget in BUDGETS:
            at_budget = selected[selected["target"] == budget]
            wide = at_budget.pivot(
                index="replication", columns="detector", values="test_detection_probability"
            ).sort_index()
            for augmented, base in AUGMENTATIONS:
                row = {
                    "family": "detection",
                    "unit": label,
                    "learner": "selected",
                    "contrast": "{} - {}".format(augmented, base),
                    "budget": budget,
                }
                row.update(paired_t(wide[augmented] - wide[base]))
                rows.append(row)
    return rows


def market_contrasts() -> List[Dict[str, object]]:
    rows = []
    for market, label in MARKETS:
        frame = pd.read_csv(MARKET_DIR / market / "summary.csv").set_index("cell")
        for augmented, base in AUGMENTATIONS:
            entry = frame.loc[augmented]
            rows.append(
                {
                    "family": "market",
                    "unit": label,
                    "learner": "selected",
                    "contrast": "{} - {}".format(augmented, base),
                    "budget": "",
                    "n": int(entry["n_observations"]),
                    "delta_points": float(entry["gain_points"]),
                    "ci_low": float(entry["ci_low"]),
                    "ci_high": float(entry["ci_high"]),
                    "p_raw": float(entry["bootstrap_p"]),
                    "method": "stationary bootstrap",
                }
            )
    return rows


def build_table() -> pd.DataFrame:
    """Every reported contrast with raw, within-family and global Holm p-values."""

    rows = classification_contrasts() + detection_contrasts() + market_contrasts()
    table = pd.DataFrame(rows)
    table["p_holm_family"] = np.nan
    for family in table["family"].unique():
        mask = table["family"] == family
        table.loc[mask, "p_holm_family"] = holm(table.loc[mask, "p_raw"])
    table["p_holm_global"] = holm(table["p_raw"])
    for column, source in (
        ("resolved_raw", "p_raw"),
        ("resolved_family", "p_holm_family"),
        ("resolved_global", "p_holm_global"),
    ):
        table[column] = table[source] < ALPHA
    return table


def report(table: pd.DataFrame) -> str:
    """One line per family, then every contrast the correction changes."""

    lines = ["Contrasts resolved at the 5% level, by family:"]
    for family in ("classification", "detection", "market"):
        part = table[table["family"] == family]
        lines.append(
            "  {:<15} {:>2} contrasts: {:>2} uncorrected, {:>2} after Holm in family, "
            "{:>2} after Holm across all {}".format(
                family,
                len(part),
                int(part["resolved_raw"].sum()),
                int(part["resolved_family"].sum()),
                int(part["resolved_global"].sum()),
                len(table),
            )
        )
    lost = table[table["resolved_raw"] & ~table["resolved_family"]]
    lines.append("\nResolved uncorrected but not after the within-family correction:")
    if lost.empty:
        lines.append("  none")
    for _, row in lost.iterrows():
        lines.append(
            "  {:<14} {:<10} {:<12} {:<35} {:+.2f} pp  p={:.4f} -> {:.4f}".format(
                row["family"],
                row["unit"],
                str(row["budget"]),
                row["contrast"],
                row["delta_points"],
                row["p_raw"],
                row["p_holm_family"],
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=Path("results/multiplicity"))
    args = parser.parse_args()
    table = build_table()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "family", "unit", "learner", "budget", "contrast", "n", "method",
        "delta_points", "ci_low", "ci_high",
        "p_raw", "p_holm_family", "p_holm_global",
        "resolved_raw", "resolved_family", "resolved_global",
    ]
    table[columns].to_csv(args.output_dir / "holm.csv", index=False)
    print(report(table))
    print("\nSaved to {}".format(args.output_dir / "holm.csv"))


if __name__ == "__main__":
    main()
