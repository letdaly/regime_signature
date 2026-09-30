"""Discretization audit of the Design B variance process (Supporting Information, Section S1).

In Design B every state shares the stationary variance law Gamma(2 theta / q, q / 2), with
mean theta and variance theta q / 2. The Euler scheme with absorption at zero used by
``regime_pipeline.HestonPathSimulator`` preserves the mean of the variance process, but
while absorption is inactive it inflates the stationary variance by the factor
1 / (1 - kappa delta / 2) for a step delta, so states with faster mean reversion drift
further from the matched law. The audit simulates the variance process of each Design B
state at the midpoints of the parameter ranges, with one and with four Euler substeps per
daily observation, and compares its stationary mean and variance with the exact values.

The update below is the variance step of ``HestonPathSimulator.simulate``, vectorized over
independent paths:

    v_plus = max(v, 0)
    v = v + kappa (theta - v_plus) delta + sigma sqrt(v_plus delta) z
    v = max(v, 0)

    python audit_discretization.py --output-dir reproduced/discretization_audit
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

THETA = 0.08
Q = 0.045
KAPPAS = (1.0, 5.25, 16.0)
SUBSTEPS = (1, 4)
DT = 1.0 / 252.0


def batch_standard_error(per_path: np.ndarray, statistic, n_batches: int) -> float:
    """Standard error of ``statistic`` from independent batches of paths."""

    values = [statistic(batch) for batch in np.array_split(per_path, n_batches)]
    return float(np.std(values, ddof=1) / np.sqrt(n_batches))


def audit_state(kappa: float, substeps: int, n_paths: int, burn_in: int, n_obs: int, seed: int, n_batches: int) -> dict:
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(Q * kappa)
    delta = DT / substeps
    v = rng.gamma(2.0 * THETA / Q, Q / 2.0, size=n_paths)
    first = np.zeros(n_paths)
    second = np.zeros(n_paths)
    absorbed = 0
    for step in range(burn_in + n_obs):
        for _ in range(substeps):
            v_plus = np.maximum(v, 0.0)
            v = v + kappa * (THETA - v_plus) * delta + sigma * np.sqrt(v_plus * delta) * rng.standard_normal(n_paths)
            absorbed += int(np.count_nonzero(v < 0.0))
            v = np.maximum(v, 0.0)
        if step >= burn_in:
            first += v
            second += v * v
    moments = np.column_stack((first / n_obs, second / n_obs))

    def mean_of(batch):
        return batch[:, 0].mean()

    def variance_of(batch):
        return batch[:, 1].mean() - batch[:, 0].mean() ** 2

    exact_variance = THETA * Q / 2.0
    mean = mean_of(moments)
    variance = variance_of(moments)
    return {
        "substeps": substeps,
        "kappa": kappa,
        "sigma": float(sigma),
        "stationary_mean": float(mean),
        "mean_deviation_pct": float(100.0 * (mean / THETA - 1.0)),
        "mean_deviation_se_pct": 100.0 * batch_standard_error(moments, mean_of, n_batches) / THETA,
        "stationary_variance": float(variance),
        "variance_deviation_pct": float(100.0 * (variance / exact_variance - 1.0)),
        "variance_deviation_se_pct": 100.0 * batch_standard_error(moments, variance_of, n_batches) / exact_variance,
        "euler_variance_deviation_pct": float(100.0 * (1.0 / (1.0 - kappa * delta / 2.0) - 1.0)),
        "absorbed_share": absorbed / float(n_paths * (burn_in + n_obs) * substeps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", type=int, default=100000)
    parser.add_argument("--burn-in", type=int, default=2520, help="Daily observations discarded (10 years).")
    parser.add_argument("--observations", type=int, default=2520, help="Daily observations averaged (10 years).")
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--output-dir", type=Path, default=Path("results/discretization_audit"))
    args = parser.parse_args()

    rows = []
    for substeps in SUBSTEPS:
        for index, kappa in enumerate(KAPPAS):
            rows.append(
                audit_state(kappa, substeps, args.paths, args.burn_in, args.observations,
                            args.seed + 100 * substeps + index, args.batches)
            )
    audit = pd.DataFrame(rows)
    spread = (
        audit.groupby("substeps")["variance_deviation_pct"].agg(lambda x: x.max() - x.min()).rename("variance_spread_pp")
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output_dir / "audit.csv", index=False)
    configuration = {
        "experiment": "discretization_audit",
        "design": "matched_marginal (Design B)",
        "theta": THETA,
        "q": Q,
        "kappas": list(KAPPAS),
        "substeps": list(SUBSTEPS),
        "dt": DT,
        "paths": args.paths,
        "burn_in_observations": args.burn_in,
        "averaged_observations": args.observations,
        "batches": args.batches,
        "seed": args.seed,
        "initial_law": "exact stationary Gamma(2 theta / q, q / 2)",
        "variance_spread_pp": {str(k): float(v) for k, v in spread.items()},
    }
    (args.output_dir / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(audit.round(4).to_string(index=False))
        print(spread.round(3).to_string())


if __name__ == "__main__":
    main()
