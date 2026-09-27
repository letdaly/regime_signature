"""Paired Monte Carlo study for the fBM-driven stochastic-volatility row.

This runner reuses the Experiment A/B regime designs (scale-only or matched
marginal), the fixed Random Forest, the causal detector, and every metric of
the standard-Heston rows.  The only change is the simulator: the CIR variance
equation is driven by fractional Gaussian noise with Hurst index ``hurst``
(``regime_pipeline.FBMStochasticVolatilitySimulator``).

Design choices that still need a decision are deliberately explicit command
line parameters with no silent default:

- ``--scenario``: which Heston regime contrast the roughness row modifies.
- ``--hurst``: the Hurst index (one run per value, e.g. 0.2 and 0.35).
- ``--sigma-scale``: multiplier applied to every regime's vol-of-vol.  With
  the default 1.0 the row uses *exactly* the Heston parameter families, but
  rough noise shrinks the stationary dispersion of the variance, so the row is
  not marginally matched to the Heston row.  A value above one can compensate;
  ``--audit-only`` quantifies the gap before anything is decided.
- ``--base-seed``: defaults to the base seed of the corresponding Heston row,
  so replication ``k`` draws the same parameter families as replication ``k``
  of that row and cross-row contrasts can be paired by seed.

Nothing here overwrites the frozen Heston results.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from experiment_a_monte_carlo import (
    DEFAULT_EVENT_TOLERANCE,
    DEFAULT_LEARNERS,
    DEFAULT_SIGNATURE_KIND,
    LOGISTIC_C_GRID,
    REPRESENTATIONS,
    add_learner_argument,
    add_path_count_arguments,
    add_signature_kind_argument,
    evaluate_splits,
    guard_output_directory,
    progress_line,
    replication_seeds,
    save_result_tables,
    summarize_results,
)
from regime_detection import DETECTOR_GRID
from regime_pipeline import (
    FBMStochasticVolatilitySimulator,
    HestonPathSimulator,
    CausalFeatureExtractor,
    ParameterSampler,
    PathLevelSplits,
    build_path_level_splits,
    sample_matched_marginal_parameters,
    sample_scale_only_parameters,
)


# Each scenario mirrors one standard-Heston row exactly: same sampler, same
# substep convention, same base seed (hence the same parameter families per
# replication), and the directory holding that row's frozen results.
SCENARIOS: Dict[str, Dict[str, object]] = {
    "scale_only": {
        "sampler": sample_scale_only_parameters,
        "substeps": 1,
        "base_seed": 20260827,
        "heston_results": "results/main_grid/heston_scale_only",
    },
    "matched_marginal": {
        "sampler": sample_matched_marginal_parameters,
        "substeps": 4,
        "base_seed": 20260828,
        "heston_results": "results/main_grid/heston_matched_marginal",
    },
}
REFERENCE_HURST = 0.5


def scaled_sampler(sampler: ParameterSampler, sigma_scale: float) -> ParameterSampler:
    """Multiply every regime's sigma by ``sigma_scale`` after sampling.

    Both scenario invariants survive: in the scale-only design sigma_i =
    c * sqrt(kappa * theta_i) becomes (c * s) * sqrt(kappa * theta_i), and in
    the matched-marginal design sigma^2 / kappa stays common to all regimes.
    """

    if sigma_scale <= 0.0:
        raise ValueError("sigma_scale must be positive.")
    if sigma_scale == 1.0:
        return sampler

    def sample(rng: np.random.Generator):
        parameters = sampler(rng)
        return {
            regime: {**values, "sigma": float(values["sigma"] * sigma_scale)}
            for regime, values in parameters.items()
        }

    return sample


def build_fbm_splits(
    seed: int,
    scenario: str,
    hurst: float,
    sigma_scale: float,
    n_paths: Tuple[int, int, int],
    n_steps: int,
    window_size: int,
    p_switch: float,
    substeps: int,
    signature_kind: str = DEFAULT_SIGNATURE_KIND,
) -> PathLevelSplits:
    """Simulate one replication's paths with the fBM-driven simulator."""

    extractor = CausalFeatureExtractor(
        window_size=window_size,
        short_window=min(10, window_size),
        signature_level=3,
        representation="combined",
        signature_kind=signature_kind,
    )
    return build_path_level_splits(
        extractor=extractor,
        parameter_sampler=scaled_sampler(SCENARIOS[scenario]["sampler"], sigma_scale),
        n_paths=n_paths,
        n_steps=n_steps,
        master_seed=seed,
        p_switch=p_switch,
        min_dwell=window_size,
        stationary_initial_variance=True,
        simulation_substeps=substeps,
        hurst=hurst,
    )


def evaluate_paired_replication(
    replication: int,
    seed: int,
    scenario: str,
    hurst: float,
    sigma_scale: float,
    n_paths: Tuple[int, int, int],
    n_steps: int,
    window_size: int,
    p_switch: float,
    n_estimators: int,
    substeps: int,
    model_jobs: int = -1,
    event_tolerance: int = DEFAULT_EVENT_TOLERANCE,
    evaluate_detector: bool = True,
    signature_kind: str = DEFAULT_SIGNATURE_KIND,
    learners: Tuple[str, ...] = DEFAULT_LEARNERS,
) -> Dict[str, float]:
    """Evaluate all representations on one shared fBM-driven path collection."""

    splits = build_fbm_splits(
        seed=seed,
        scenario=scenario,
        hurst=hurst,
        sigma_scale=sigma_scale,
        n_paths=n_paths,
        n_steps=n_steps,
        window_size=window_size,
        p_switch=p_switch,
        substeps=substeps,
        signature_kind=signature_kind,
    )
    row = evaluate_splits(
        splits,
        replication=replication,
        seed=seed,
        n_estimators=n_estimators,
        model_jobs=model_jobs,
        event_tolerance=event_tolerance,
        evaluate_detector=evaluate_detector,
        learners=learners,
    )
    row["hurst"] = float(hurst)
    row["sigma_scale"] = float(sigma_scale)
    return row


def _lag_one_autocorrelation(values: np.ndarray) -> float:
    centered = values - values.mean()
    denominator = float(np.dot(centered, centered))
    return float(np.dot(centered[:-1], centered[1:]) / denominator) if denominator else np.nan


def stationary_variance_audit(
    scenario: str,
    hurst: float,
    sigma_scale: float,
    seed: int,
    n_steps: int,
    substeps: int,
) -> pd.DataFrame:
    """Compare long no-switch paths under the rough setting and the Heston reference.

    One parameter family is drawn from ``seed``.  For every regime the variance
    and return moments are measured on a single long path (``n_steps``
    observations, stationary initialization) under (hurst, sigma_scale) and
    under the Brownian reference (H = 0.5, sigma_scale = 1), which is the
    setting of the corresponding standard-Heston row.  The CIR stationary
    standard deviation sqrt(sigma^2 theta / (2 kappa)) is reported as the
    theoretical H = 0.5 value.
    """

    family_rng = np.random.default_rng(seed)
    parameters = scaled_sampler(SCENARIOS[scenario]["sampler"], sigma_scale)(family_rng)
    reference_parameters = SCENARIOS[scenario]["sampler"](np.random.default_rng(seed))
    settings = (
        ("rough", hurst, sigma_scale, parameters),
        ("heston_reference", REFERENCE_HURST, 1.0, reference_parameters),
    )
    rows = []
    for label, setting_hurst, setting_scale, family in settings:
        for regime, values in family.items():
            path_seed = int(np.random.default_rng([seed, regime]).integers(0, 2 ** 32 - 1))
            if setting_hurst == REFERENCE_HURST:
                simulator = HestonPathSimulator(substeps=substeps)
            else:
                simulator = FBMStochasticVolatilitySimulator(
                    hurst=setting_hurst, substeps=substeps
                )
            frame = simulator.simulate(
                path_id="audit_{}_{}".format(label, regime),
                regime_parameters={0: values},
                n_steps=n_steps,
                seed=path_seed,
                p_switch=0.0,
                v0=None,
            ).frame
            variance = frame["Variance"].to_numpy()
            returns = frame["LogReturn"].to_numpy()[1:]
            rows.append(
                {
                    "setting": label,
                    "hurst": setting_hurst,
                    "sigma_scale": setting_scale,
                    "regime": int(regime),
                    "kappa": values["kappa"],
                    "theta": values["theta"],
                    "sigma": values["sigma"],
                    "feller_ratio": 2.0 * values["kappa"] * values["theta"] / values["sigma"] ** 2,
                    "cir_stationary_sd": float(
                        np.sqrt(values["sigma"] ** 2 * values["theta"] / (2.0 * values["kappa"]))
                    ),
                    "variance_mean": float(variance.mean()),
                    "variance_sd": float(variance.std(ddof=1)),
                    "variance_zero_fraction": float(np.mean(variance <= 0.0)),
                    "variance_increment_lag1_autocorr": _lag_one_autocorrelation(np.diff(variance)),
                    "return_sd_annualized": float(returns.std(ddof=1) * np.sqrt(252.0)),
                    "return_excess_kurtosis": float(kurtosis(returns, fisher=True, bias=False)),
                    "return_lag1_autocorr": _lag_one_autocorrelation(returns),
                    "abs_return_lag1_autocorr": _lag_one_autocorrelation(np.abs(returns)),
                    "n_steps": int(n_steps),
                    "substeps": int(substeps),
                }
            )
    audit = pd.DataFrame(rows)
    reference = audit[audit["setting"] == "heston_reference"].set_index("regime")
    rough = audit["setting"] == "rough"
    audit.loc[rough, "variance_sd_ratio_to_heston"] = (
        audit.loc[rough, "variance_sd"].to_numpy()
        / reference.loc[audit.loc[rough, "regime"], "variance_sd"].to_numpy()
    )
    audit.loc[rough, "return_sd_ratio_to_heston"] = (
        audit.loc[rough, "return_sd_annualized"].to_numpy()
        / reference.loc[audit.loc[rough, "regime"], "return_sd_annualized"].to_numpy()
    )
    return audit


def _evaluate_task(task: Dict[str, object]) -> Dict[str, float]:
    started = time.monotonic()
    row = evaluate_paired_replication(**task)
    row["elapsed_seconds"] = float(time.monotonic() - started)
    return row


def default_output_dir(scenario: str, hurst: float, sigma_scale: float) -> Path:
    name = "fbm_{}_h{:g}".format(scenario, hurst)
    if sigma_scale != 1.0:
        name += "_sigma{:g}".format(sigma_scale)
    return Path("results") / "main_grid" / name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--scenario",
        required=True,
        choices=sorted(SCENARIOS),
        help="Heston regime contrast to run under rough variance noise.",
    )
    parser.add_argument(
        "--hurst", type=float, required=True, help="Hurst index in (0, 1), e.g. 0.2 or 0.35."
    )
    parser.add_argument(
        "--sigma-scale",
        type=float,
        default=1.0,
        help="Multiplier on every regime's sigma (1.0 = identical Heston parameter families).",
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=None,
        help="Simulation substeps per observation; defaults to the scenario's Heston convention.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help="Defaults to the matching Heston row's base seed (paired parameter draws).",
    )
    parser.add_argument("--replications", type=int, default=50)
    add_path_count_arguments(parser)
    add_signature_kind_argument(parser)
    add_learner_argument(parser)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--p-switch", type=float, default=0.002)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--event-tolerance", type=int, default=DEFAULT_EVENT_TOLERANCE)
    parser.add_argument("--skip-detector", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to results/main_grid/fbm_<scenario>_h<hurst>[_sigma<scale>].",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Write variance_audit.csv comparing the rough and Heston settings, then exit.",
    )
    parser.add_argument("--audit-steps", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.hurst < 1.0:
        raise ValueError("hurst must lie in (0, 1).")
    if args.replications < 2:
        raise ValueError("At least two replications are required.")
    if min(args.train_paths, args.validation_paths, args.test_paths) < 1:
        raise ValueError("Every split needs at least one path.")
    if args.jobs < 1 or args.event_tolerance < 1 or args.audit_steps < 100:
        raise ValueError("jobs, event-tolerance, and audit-steps must be positive.")

    scenario = SCENARIOS[args.scenario]
    substeps = int(scenario["substeps"] if args.substeps is None else args.substeps)
    base_seed = int(scenario["base_seed"] if args.base_seed is None else args.base_seed)
    output_dir = (
        default_output_dir(args.scenario, args.hurst, args.sigma_scale)
        if args.output_dir is None
        else args.output_dir
    ).resolve()
    if not args.audit_only:
        guard_output_directory(output_dir, args.resume)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "row": "fbm_driven_stochastic_volatility",
        "simulator": "FBMStochasticVolatilitySimulator (exact Hosking fGn, innovation-correlated price)",
        "scenario": args.scenario,
        "hurst": args.hurst,
        "sigma_scale": args.sigma_scale,
        "simulation_substeps": substeps,
        "base_seed": base_seed,
        "paired_parameter_draws_with": scenario["heston_results"]
        if base_seed == scenario["base_seed"]
        else None,
        "replications": args.replications,
        "n_paths": [args.train_paths, args.validation_paths, args.test_paths],
        "n_steps": args.steps,
        "window_size": args.window,
        "p_switch": args.p_switch,
        "n_estimators": args.trees,
        "signature_kind": args.signature_kind,
        "signature_level": 3,
        "signature_path": CausalFeatureExtractor.PATH_CONVENTION,
        "learners": list(args.learners),
        "logistic_c_grid": list(LOGISTIC_C_GRID),
        "stationary_initial_variance": True,
        "stationary_initial_variance_note": "CIR Gamma law; exact only for hurst = 0.5",
        "evaluate_detector": not args.skip_detector,
        "event_tolerance": args.event_tolerance,
        "detector_grid": [list(values) for values in DETECTOR_GRID],
    }

    audit_path = output_dir / "variance_audit.csv"
    audit = stationary_variance_audit(
        scenario=args.scenario,
        hurst=args.hurst,
        sigma_scale=args.sigma_scale,
        seed=replication_seeds(base_seed, 1)[0],
        n_steps=args.audit_steps,
        substeps=substeps,
    )
    audit.to_csv(audit_path, index=False)
    display = audit[
        [
            "setting", "regime", "sigma", "variance_mean", "variance_sd",
            "cir_stationary_sd", "variance_sd_ratio_to_heston",
            "return_sd_annualized", "return_excess_kurtosis",
            "variance_increment_lag1_autocorr",
        ]
    ]
    print("Stationary variance audit (one parameter family, no switches):")
    print(display.to_string(index=False, float_format=lambda value: "{:.4f}".format(value)))
    print("Saved audit: {}".format(audit_path))
    if args.audit_only:
        return

    config_path = output_dir / "config.json"
    if args.resume and config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Cannot resume: current arguments differ from config.json.")
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    replications_path = output_dir / "replications.csv"
    summary_path = output_dir / "summary.csv"
    class_counts_path = output_dir / "class_counts.csv"
    diagnostics_path = output_dir / "test_diagnostics.csv"
    if args.resume and replications_path.exists():
        existing = pd.read_csv(replications_path)
        completed = set(existing["replication"].astype(int).tolist())
        rows = existing.to_dict("records")
    else:
        completed = set()
        rows = []

    seeds = replication_seeds(base_seed, args.replications)
    tasks = [
        {
            "replication": replication,
            "seed": seed,
            "scenario": args.scenario,
            "hurst": args.hurst,
            "sigma_scale": args.sigma_scale,
            "n_paths": (args.train_paths, args.validation_paths, args.test_paths),
            "n_steps": args.steps,
            "window_size": args.window,
            "p_switch": args.p_switch,
            "n_estimators": args.trees,
            "substeps": substeps,
            "model_jobs": -1 if args.jobs == 1 else 1,
            "event_tolerance": args.event_tolerance,
            "evaluate_detector": not args.skip_detector,
            "signature_kind": args.signature_kind,
            "learners": tuple(args.learners),
        }
        for replication, seed in enumerate(seeds, start=1)
        if replication not in completed
    ]

    def checkpoint(row: Dict[str, float]) -> None:
        rows.append(row)
        current = pd.DataFrame(rows).sort_values("replication").reset_index(drop=True)
        save_result_tables(
            current, replications_path, summary_path, class_counts_path, diagnostics_path
        )
        detector_line = ""
        if "combined_test_boundary_f1" in row:
            detector_line = " boundary-F1={:.3f}/{:.3f}/{:.3f} delay={:.1f}/{:.1f}/{:.1f}".format(
                *[
                    row[representation + "_test_" + metric]
                    for metric in ("boundary_f1", "mean_detection_delay")
                    for representation in REPRESENTATIONS
                ]
            )
        print(
            "Replication {}/{} seed={} H={:g} {}{} ({:.1f}s)".format(
                int(row["replication"]),
                args.replications,
                int(row["seed"]),
                args.hurst,
                progress_line(row, args.learners),
                detector_line,
                row["elapsed_seconds"],
            ),
            flush=True,
        )

    started = time.monotonic()
    if args.jobs == 1:
        for task in tasks:
            checkpoint(_evaluate_task(task))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(_evaluate_task, task) for task in tasks]
            for future in as_completed(futures):
                checkpoint(future.result())

    results = pd.DataFrame(rows).sort_values("replication").reset_index(drop=True)
    save_result_tables(
        results, replications_path, summary_path, class_counts_path, diagnostics_path
    )
    print("\nPaired fBM-driven SV summary ({} , H={:g}, sigma_scale={:g})".format(
        args.scenario, args.hurst, args.sigma_scale
    ))
    print(
        summarize_results(results).to_string(
            index=False, float_format=lambda value: "{:.6f}".format(value)
        )
    )
    print("\nSaved results to {}".format(output_dir))
    print("Total time: {:.1f}s".format(time.monotonic() - started))


if __name__ == "__main__":
    main()
