"""Leakage-free data generation and feature extraction for regime detection.

The key unit of independence is a complete simulated path.  Rolling windows may
overlap within a path, as they will during deployment, but a path is assigned to
exactly one of train, validation, or test before any windows are extracted.
"""

from dataclasses import dataclass
import zlib
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import iisignature
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


PARAMETER_NAMES = ("mu", "kappa", "theta", "sigma", "rho")
# Train / validation / test paths per replication for the main-table grid.
# Validation was raised from 3 to 10 paths on 2026-09-17: three paths were too
# few to select among the 27 causal-detector settings.  The frozen results in
# results/experiment_a_scale_only*, results/experiment_b_diagnostics, and
# results/experiment_b_window_stability were produced with (10, 3, 5) and are
# not comparable cell-for-cell.
DEFAULT_N_PATHS = (10, 10, 5)
# Optional per-regime Hurst index.  Only FBMStochasticVolatilitySimulator
# consumes it; HestonPathSimulator rejects it so a roughness-shift design
# cannot silently degrade into a standard Heston run.
OPTIONAL_PARAMETER_NAMES = ("hurst",)
ParameterValues = Dict[str, float]
ParameterRanges = Mapping[int, Mapping[str, Tuple[float, float]]]
ParameterSampler = Callable[[np.random.Generator], Dict[int, ParameterValues]]


# These ranges define volatility-level-separated regimes.  Each independent
# path receives a fresh draw for every regime instead of reusing three fixed
# parameter fingerprints across the whole experiment.
DEFAULT_LEVEL_SHIFT_RANGES: Dict[int, Dict[str, Tuple[float, float]]] = {
    0: {
        "mu": (0.03, 0.07),
        "kappa": (4.5, 5.5),
        "theta": (0.015, 0.025),
        "sigma": (0.12, 0.20),
        "rho": (-0.75, -0.65),
    },
    1: {
        "mu": (-0.01, 0.03),
        "kappa": (3.2, 4.2),
        "theta": (0.065, 0.095),
        "sigma": (0.22, 0.32),
        "rho": (-0.80, -0.70),
    },
    2: {
        "mu": (-0.10, -0.04),
        "kappa": (1.8, 3.0),
        "theta": (0.14, 0.20),
        "sigma": (0.32, 0.48),
        "rho": (-0.90, -0.78),
    },
}


# Experiment A is a pure scale experiment.  Within each independently sampled
# path, mu, kappa, rho, and c are shared by all regimes.  Only theta changes,
# and sigma_i = c * sqrt(kappa * theta_i).  Consequently, away from switching
# boundaries, X_t = V_t / theta_i follows the same CIR dynamics in every regime:
#
#   dX_t = kappa * (1 - X_t) dt + c * sqrt(kappa * X_t) dW_t.
SCALE_ONLY_THETA_RANGES: Dict[int, Tuple[float, float]] = {
    0: (0.015, 0.025),
    1: (0.065, 0.095),
    2: (0.14, 0.20),
}


def sample_scale_only_parameters(
    rng: np.random.Generator,
) -> Dict[int, ParameterValues]:
    """Draw one Experiment A family whose regimes differ only by scale."""

    mu = float(rng.uniform(0.01, 0.04))
    kappa = float(rng.uniform(3.5, 4.5))
    rho = float(rng.uniform(-0.80, -0.65))
    scale_coefficient = float(rng.uniform(0.45, 0.55))

    sampled: Dict[int, ParameterValues] = {}
    for regime, (low, high) in SCALE_ONLY_THETA_RANGES.items():
        theta = float(rng.uniform(low, high))
        sampled[regime] = {
            "mu": mu,
            "kappa": kappa,
            "theta": theta,
            "sigma": float(scale_coefficient * np.sqrt(kappa * theta)),
            "rho": rho,
        }
    return sampled


# Experiment B separates temporal dependence without changing the stationary
# one-time law of the CIR variance process.  For every path, theta and
# sigma^2/kappa are common to all regimes; only kappa is drawn from these
# non-overlapping ranges.  Because the stationary CIR distribution is Gamma
# with shape 2*kappa*theta/sigma^2 and scale sigma^2/(2*kappa), both its mean
# and its complete distribution are matched across the three regimes.
MATCHED_MARGINAL_KAPPA_RANGES: Dict[int, Tuple[float, float]] = {
    0: (0.8, 1.2),
    1: (4.5, 6.0),
    2: (14.0, 18.0),
}


def sample_matched_marginal_parameters(
    rng: np.random.Generator,
) -> Dict[int, ParameterValues]:
    """Draw one Experiment B family with an exactly matched CIR marginal law."""

    theta = float(rng.uniform(0.07, 0.09))
    variance_ratio = float(rng.uniform(0.035, 0.055))
    mu = float(rng.uniform(0.01, 0.04))
    rho = float(rng.uniform(-0.80, -0.65))

    sampled: Dict[int, ParameterValues] = {}
    for regime, (low, high) in MATCHED_MARGINAL_KAPPA_RANGES.items():
        kappa = float(rng.uniform(low, high))
        sampled[regime] = {
            "mu": mu,
            "kappa": kappa,
            "theta": theta,
            "sigma": float(np.sqrt(variance_ratio * kappa)),
            "rho": rho,
        }
    return sampled


@dataclass(frozen=True)
class SimulatedPath:
    """One independent trajectory and the parameters that generated it."""

    path_id: str
    frame: pd.DataFrame
    regime_parameters: Dict[int, ParameterValues]
    seed: int


@dataclass(frozen=True)
class WindowDataset:
    """Window features with provenance needed for leakage audits."""

    X: np.ndarray
    y: np.ndarray
    path_ids: np.ndarray
    end_times: np.ndarray
    feature_names: Tuple[str, ...]

    def __len__(self) -> int:
        return len(self.y)


@dataclass(frozen=True)
class PathLevelSplits:
    """Datasets created only after complete paths are assigned to partitions."""

    train: WindowDataset
    validation: WindowDataset
    test: WindowDataset
    paths: Dict[str, Tuple[SimulatedPath, ...]]

    def assert_disjoint(self) -> None:
        declared_ids = {
            split_name: [path.path_id for path in self.paths[split_name]]
            for split_name in ("train", "validation", "test")
        }
        path_sets = {
            "train": set(self.train.path_ids.tolist()),
            "validation": set(self.validation.path_ids.tolist()),
            "test": set(self.test.path_ids.tolist()),
        }
        for split_name in ("train", "validation", "test"):
            if len(declared_ids[split_name]) != len(set(declared_ids[split_name])):
                raise AssertionError(
                    "Duplicate declared path IDs in {}.".format(split_name)
                )
            if path_sets[split_name] != set(declared_ids[split_name]):
                raise AssertionError(
                    "Window provenance does not match declared {} paths.".format(
                        split_name
                    )
                )
        pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
        for left, right in pairs:
            overlap = path_sets[left] & path_sets[right]
            if overlap:
                raise AssertionError(
                    "Path leakage between {} and {}: {}".format(left, right, sorted(overlap))
                )

        # A renamed path must not evade the provenance audit. Simulation seeds
        # are unique experiment-wide, and the same SimulatedPath object cannot
        # be declared in two partitions.
        all_paths = [
            path
            for split_name in ("train", "validation", "test")
            for path in self.paths[split_name]
        ]
        seeds = [path.seed for path in all_paths]
        if len(seeds) != len(set(seeds)):
            raise AssertionError("Simulation seed reused across declared paths.")
        identities = [id(path) for path in all_paths]
        if len(identities) != len(set(identities)):
            raise AssertionError("SimulatedPath object reused across partitions.")


def _validate_parameter_ranges(parameter_ranges: ParameterRanges) -> None:
    if len(parameter_ranges) < 2:
        raise ValueError("At least two regimes are required.")
    expected_ids = set(range(len(parameter_ranges)))
    if set(parameter_ranges) != expected_ids:
        raise ValueError("Regime IDs must be consecutive integers beginning at zero.")
    for regime, ranges in parameter_ranges.items():
        if set(ranges) - set(OPTIONAL_PARAMETER_NAMES) != set(PARAMETER_NAMES):
            raise ValueError(
                "Regime {} must define {} (optionally {}).".format(
                    regime, PARAMETER_NAMES, OPTIONAL_PARAMETER_NAMES
                )
            )
        for name, bounds in ranges.items():
            low, high = bounds
            if not np.isfinite(low) or not np.isfinite(high) or low > high:
                raise ValueError("Invalid range for regime {} parameter {}.".format(regime, name))
        if not -1.0 <= ranges["rho"][0] <= ranges["rho"][1] <= 1.0:
            raise ValueError("rho must stay in [-1, 1].")
        if "hurst" in ranges and not 0.0 < ranges["hurst"][0] <= ranges["hurst"][1] < 1.0:
            raise ValueError("hurst must stay in the open interval (0, 1).")


def sample_regime_parameters(
    parameter_ranges: ParameterRanges, rng: np.random.Generator
) -> Dict[int, ParameterValues]:
    """Draw one parameter tuple per regime for a single independent path."""

    _validate_parameter_ranges(parameter_ranges)
    sampled: Dict[int, ParameterValues] = {}
    for regime, ranges in parameter_ranges.items():
        sampled[regime] = {
            name: float(rng.uniform(low, high)) if high > low else float(low)
            for name, (low, high) in ranges.items()
        }
    return sampled


def _initial_variance(
    rng: np.random.Generator,
    parameters: Mapping[str, float],
    v0: Optional[float],
) -> float:
    """Return v0, or one draw from the CIR stationary Gamma law when v0 is None."""

    if v0 is not None:
        return float(v0)
    kappa = float(parameters["kappa"])
    theta = float(parameters["theta"])
    sigma = float(parameters["sigma"])
    if kappa <= 0.0 or theta <= 0.0 or sigma <= 0.0:
        raise ValueError("Stationary variance initialization needs positive CIR parameters.")
    stationary_shape = 2.0 * kappa * theta / sigma ** 2
    stationary_scale = sigma ** 2 / (2.0 * kappa)
    return float(rng.gamma(stationary_shape, stationary_scale))


class HestonPathSimulator:
    """Standard Heston simulator with local RNG state and persistent regimes."""

    def __init__(self, dt: float = 1.0 / 252.0, substeps: int = 1):
        if dt <= 0:
            raise ValueError("dt must be positive.")
        if substeps < 1:
            raise ValueError("substeps must be positive.")
        self.dt = dt
        self.substeps = int(substeps)

    def simulate(
        self,
        path_id: str,
        regime_parameters: Mapping[int, Mapping[str, float]],
        n_steps: int,
        seed: int,
        p_switch: float = 0.002,
        min_dwell: int = 50,
        initial_regime: Optional[int] = None,
        S0: float = 100.0,
        v0: Optional[float] = 0.04,
    ) -> SimulatedPath:
        if n_steps < 2:
            raise ValueError("n_steps must be at least two.")
        if not 0.0 <= p_switch <= 1.0:
            raise ValueError("p_switch must lie in [0, 1].")
        if min_dwell < 1:
            raise ValueError("min_dwell must be positive.")

        regime_ids = tuple(sorted(regime_parameters))
        if regime_ids != tuple(range(len(regime_ids))):
            raise ValueError("Regime IDs must be consecutive integers beginning at zero.")
        if any("hurst" in regime_parameters[regime] for regime in regime_ids):
            raise ValueError(
                "HestonPathSimulator ignores roughness; use "
                "FBMStochasticVolatilitySimulator for a hurst parameter."
            )

        rng = np.random.default_rng(seed)
        current_regime = int(regime_ids[0] if initial_regime is None else initial_regime)
        if current_regime not in regime_parameters:
            raise ValueError("initial_regime is not present in regime_parameters.")

        prices = np.empty(n_steps, dtype=float)
        variances = np.empty(n_steps, dtype=float)
        regimes = np.empty(n_steps, dtype=int)
        v = _initial_variance(rng, regime_parameters[current_regime], v0)
        S = float(S0)
        dwell = 0

        for t in range(n_steps):
            if dwell >= min_dwell and rng.random() < p_switch:
                alternatives = [regime for regime in regime_ids if regime != current_regime]
                current_regime = int(rng.choice(alternatives))
                dwell = 0

            params = regime_parameters[current_regime]
            substep_dt = self.dt / self.substeps
            for _ in range(self.substeps):
                z_variance = rng.normal()
                z_independent = rng.normal()
                rho = float(params["rho"])
                z_price = (
                    rho * z_variance
                    + np.sqrt(max(1.0 - rho ** 2, 0.0)) * z_independent
                )

                v_positive = max(v, 0.0)
                v = (
                    v
                    + params["kappa"] * (params["theta"] - v_positive) * substep_dt
                    + params["sigma"] * np.sqrt(v_positive * substep_dt) * z_variance
                )
                v = max(float(v), 0.0)
                S *= np.exp(
                    (params["mu"] - 0.5 * v_positive) * substep_dt
                    + np.sqrt(v_positive * substep_dt) * z_price
                )

            prices[t] = S
            variances[t] = v
            regimes[t] = current_regime
            dwell += 1

        log_returns = np.empty(n_steps, dtype=float)
        log_returns[0] = 0.0
        log_returns[1:] = np.diff(np.log(prices))
        frame = pd.DataFrame(
            {
                "Price": prices,
                "Variance": variances,
                "LogReturn": log_returns,
                "Regime": regimes,
            }
        )
        copied_parameters = {
            regime: {name: float(value) for name, value in params.items()}
            for regime, params in regime_parameters.items()
        }
        return SimulatedPath(path_id, frame, copied_parameters, int(seed))


def _validate_hurst(hurst: float) -> float:
    hurst = float(hurst)
    if not np.isfinite(hurst) or not 0.0 < hurst < 1.0:
        raise ValueError("hurst must lie in the open interval (0, 1).")
    return hurst


def fgn_autocovariance(n_lags: int, hurst: float) -> np.ndarray:
    """Autocovariance of unit-variance fractional Gaussian noise at lags 0..n_lags.

    gamma(k) = 0.5 * (|k + 1|^{2H} - 2 |k|^{2H} + |k - 1|^{2H}).  For H = 0.5
    every lag beyond zero is exactly zero, i.e. the noise is white.
    """

    hurst = _validate_hurst(hurst)
    if n_lags < 0:
        raise ValueError("n_lags must be non-negative.")
    lags = np.arange(n_lags + 1, dtype=float)
    exponent = 2.0 * hurst
    return 0.5 * (
        np.abs(lags + 1.0) ** exponent
        - 2.0 * lags ** exponent
        + np.abs(lags - 1.0) ** exponent
    )


def fractional_gaussian_noise(white_noise: np.ndarray, hurst: float) -> np.ndarray:
    """Map i.i.d. standard normals causally onto unit-variance fGn (Hosking, 1984).

    The recursion is the Levinson-Durbin innovations algorithm: element ``t``
    of the output is the conditional mean given elements ``0..t-1`` plus the
    conditional standard deviation times ``white_noise[t]``.  Consequently the
    output depends only on ``white_noise[:t + 1]`` (it is causal), the sample is
    exact rather than approximate, and ``white_noise[t]`` is precisely the
    innovation of the fGn at step ``t``.  For ``hurst == 0.5`` the map is the
    identity.

    Cost is O(n^2) time and O(n) memory, well under a second for n = 20,000.
    The legacy notebook used an O(n^3) Cholesky factorization that silently
    fell back to ordinary Brownian motion whenever it failed.
    """

    white_noise = np.asarray(white_noise, dtype=float)
    if white_noise.ndim != 1 or white_noise.size == 0:
        raise ValueError("white_noise must be a non-empty one-dimensional array.")
    hurst = _validate_hurst(hurst)
    n = white_noise.size
    if hurst == 0.5:
        return white_noise.copy()

    gamma = fgn_autocovariance(n, hurst)
    noise = np.empty(n, dtype=float)
    noise[0] = white_noise[0]
    # phi[j - 1] multiplies noise[t - j] in the one-step-ahead predictor.
    phi = np.zeros(n, dtype=float)
    innovation_variance = 1.0
    for t in range(1, n):
        previous = phi[: t - 1]
        partial = (gamma[t] - np.dot(previous, gamma[t - 1:0:-1])) / innovation_variance
        phi[: t - 1] = previous - partial * previous[::-1]
        phi[t - 1] = partial
        innovation_variance *= 1.0 - partial * partial
        if innovation_variance <= 0.0:
            raise FloatingPointError(
                "Levinson-Durbin recursion lost positive definiteness at lag {}.".format(t)
            )
        noise[t] = np.dot(phi[:t], noise[t - 1::-1]) + np.sqrt(innovation_variance) * white_noise[t]
    return noise


class FBMStochasticVolatilitySimulator:
    """Heston-type volatility whose variance noise is fractional Gaussian noise.

    This is the *fBM-driven stochastic volatility* approximation, not rough
    Heston.  The CIR drift ``kappa * (theta - v)`` and diffusion
    ``sigma * sqrt(v)`` are unchanged; only the Brownian increments of the
    variance equation are replaced by unit-variance fractional Gaussian noise
    with Hurst index ``hurst`` on the substep grid.  Known limitations, to be
    stated in the paper:

    - The variance is not a Volterra equation, so the mean reversion is not
      fractionally integrated as in rough Heston (El Euch and Rosenbaum, 2019).
    - When ``v0`` is None the initial variance is drawn from the CIR stationary
      Gamma law, which is exact only for ``hurst == 0.5``; for other values it
      is a warm start whose transient is short relative to the path.
    - Roughness is imposed at the substep resolution.  Aggregating fGn over
      fixed blocks yields fGn with the same Hurst index, so the observed-return
      resolution inherits ``hurst`` regardless of ``substeps``.

    The price shock is ``rho * z + sqrt(1 - rho^2) * z_perp`` where ``z`` is the
    white-noise *innovation* of the variance driver, mirroring rough Heston in
    which price and variance share one Brownian motion.  Log returns therefore
    remain serially uncorrelated.  Correlating the price with the fGn itself,
    as the legacy notebook did, would give returns the spurious autocorrelation
    ``rho^2 * gamma_H(k)`` (about -0.17 at lag one for H = 0.2, rho = -0.7).

    With ``hurst == 0.5`` the recursion is exactly the Euler scheme of
    ``HestonPathSimulator``; only the order of random draws differs.
    """

    def __init__(self, hurst: float = 0.2, dt: float = 1.0 / 252.0, substeps: int = 1):
        if dt <= 0:
            raise ValueError("dt must be positive.")
        if substeps < 1:
            raise ValueError("substeps must be positive.")
        self.hurst = _validate_hurst(hurst)
        self.dt = dt
        self.substeps = int(substeps)

    @staticmethod
    def _regime_sequence(
        rng: np.random.Generator,
        regime_ids: Tuple[int, ...],
        initial_regime: int,
        n_steps: int,
        p_switch: float,
        min_dwell: int,
    ) -> np.ndarray:
        regimes = np.empty(n_steps, dtype=int)
        current_regime = initial_regime
        dwell = 0
        for t in range(n_steps):
            if dwell >= min_dwell and rng.random() < p_switch:
                alternatives = [regime for regime in regime_ids if regime != current_regime]
                current_regime = int(rng.choice(alternatives))
                dwell = 0
            regimes[t] = current_regime
            dwell += 1
        return regimes

    def simulate(
        self,
        path_id: str,
        regime_parameters: Mapping[int, Mapping[str, float]],
        n_steps: int,
        seed: int,
        p_switch: float = 0.002,
        min_dwell: int = 50,
        initial_regime: Optional[int] = None,
        S0: float = 100.0,
        v0: Optional[float] = 0.04,
    ) -> SimulatedPath:
        if n_steps < 2:
            raise ValueError("n_steps must be at least two.")
        if not 0.0 <= p_switch <= 1.0:
            raise ValueError("p_switch must lie in [0, 1].")
        if min_dwell < 1:
            raise ValueError("min_dwell must be positive.")

        regime_ids = tuple(sorted(regime_parameters))
        if regime_ids != tuple(range(len(regime_ids))):
            raise ValueError("Regime IDs must be consecutive integers beginning at zero.")
        current_regime = int(regime_ids[0] if initial_regime is None else initial_regime)
        if current_regime not in regime_parameters:
            raise ValueError("initial_regime is not present in regime_parameters.")

        # A regime may override the simulator-level Hurst index (roughness-shift
        # designs); otherwise every regime shares self.hurst.
        hurst_by_regime = {
            regime: _validate_hurst(regime_parameters[regime].get("hurst", self.hurst))
            for regime in regime_ids
        }

        rng = np.random.default_rng(seed)
        v = _initial_variance(rng, regime_parameters[current_regime], v0)
        regimes = self._regime_sequence(
            rng, regime_ids, current_regime, n_steps, p_switch, min_dwell
        )
        total_substeps = n_steps * self.substeps
        innovations = rng.standard_normal(total_substeps)
        independent = rng.standard_normal(total_substeps)
        # One exact fGn sample per distinct Hurst index, all driven by the same
        # innovations, so a roughness switch changes only the memory of the
        # variance driver and not the underlying random draws.
        drivers = {
            hurst: fractional_gaussian_noise(innovations, hurst)
            for hurst in set(hurst_by_regime.values())
        }

        prices = np.empty(n_steps, dtype=float)
        variances = np.empty(n_steps, dtype=float)
        S = float(S0)
        substep_dt = self.dt / self.substeps

        for t in range(n_steps):
            regime = int(regimes[t])
            params = regime_parameters[regime]
            driver = drivers[hurst_by_regime[regime]]
            rho = float(params["rho"])
            rho_perp = np.sqrt(max(1.0 - rho ** 2, 0.0))
            for index in range(t * self.substeps, (t + 1) * self.substeps):
                z_price = rho * innovations[index] + rho_perp * independent[index]

                v_positive = max(v, 0.0)
                v = (
                    v
                    + params["kappa"] * (params["theta"] - v_positive) * substep_dt
                    + params["sigma"] * np.sqrt(v_positive * substep_dt) * driver[index]
                )
                v = max(float(v), 0.0)
                S *= np.exp(
                    (params["mu"] - 0.5 * v_positive) * substep_dt
                    + np.sqrt(v_positive * substep_dt) * z_price
                )

            prices[t] = S
            variances[t] = v

        log_returns = np.empty(n_steps, dtype=float)
        log_returns[0] = 0.0
        log_returns[1:] = np.diff(np.log(prices))
        frame = pd.DataFrame(
            {
                "Price": prices,
                "Variance": variances,
                "LogReturn": log_returns,
                "Regime": regimes,
            }
        )
        # Record the effective Hurst index so parameter_audit_table shows it.
        copied_parameters = {
            regime: {
                **{name: float(value) for name, value in params.items()},
                "hurst": hurst_by_regime[regime],
            }
            for regime, params in regime_parameters.items()
        }
        return SimulatedPath(path_id, frame, copied_parameters, int(seed))


class CausalFeatureExtractor:
    """Extract observable, backward-looking features and endpoint-state labels."""

    REPRESENTATIONS = ("statistics", "signature", "combined")
    # The signature path of a W-return window is anchored at the origin and
    # has W increments: x_j = (j/W, sum_{a<=j} r_a, sum_{a<=j} |r_a|),
    # j = 0..W.  Before 2026-09-19 the path had W points starting at
    # (0, r_1, |r_1|); by translation invariance r_1 never reached the
    # signature.  Result directories record which convention produced them.
    PATH_CONVENTION = "origin_anchored_w_increments"
    SIGNATURE_KINDS = ("raw", "log")
    STATISTICS_LEVELS = ("legacy", "rich")
    INCREMENT_ORDERS = ("original", "shuffle", "block_shuffle")

    def __init__(
        self,
        window_size: int = 50,
        short_window: int = 10,
        signature_level: int = 3,
        representation: str = "combined",
        signature_kind: str = "raw",
        three_channel_path: bool = True,
        stride: int = 1,
        statistics_level: str = "legacy",
        increment_order: str = "original",
        order_seed: int = 0,
        block_size: int = 5,
    ):
        if window_size < 3:
            raise ValueError("window_size must be at least three.")
        if not 2 <= short_window <= window_size:
            raise ValueError("short_window must lie between two and window_size.")
        if signature_level < 1:
            raise ValueError("signature_level must be positive.")
        if representation not in self.REPRESENTATIONS:
            raise ValueError("representation must be one of {}.".format(self.REPRESENTATIONS))
        if signature_kind not in self.SIGNATURE_KINDS:
            raise ValueError("signature_kind must be one of {}.".format(self.SIGNATURE_KINDS))
        if stride < 1:
            raise ValueError("stride must be positive.")
        if statistics_level not in self.STATISTICS_LEVELS:
            raise ValueError("statistics_level must be one of {}.".format(self.STATISTICS_LEVELS))
        if increment_order not in self.INCREMENT_ORDERS:
            raise ValueError("increment_order must be one of {}.".format(self.INCREMENT_ORDERS))
        if block_size < 1:
            raise ValueError("block_size must be positive.")

        self.window_size = window_size
        self.short_window = short_window
        self.signature_level = signature_level
        self.representation = representation
        self.signature_kind = signature_kind
        self.three_channel_path = three_channel_path
        self.stride = stride
        self.statistics_level = statistics_level
        self.increment_order = increment_order
        self.order_seed = int(order_seed)
        self.block_size = int(block_size)
        self.path_dimension = 3 if three_channel_path else 2
        self._logsignature_spec = None
        if representation != "statistics" and signature_kind == "log":
            try:
                self._logsignature_spec = iisignature.prepare(
                    self.path_dimension, signature_level
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    "Logsignature preparation failed. The installed iisignature 0.24 "
                    "binary is incompatible with this environment's NumPy build. Use "
                    "signature_kind='raw' until iisignature is rebuilt; the leakage "
                    "corrections themselves do not depend on this choice."
                ) from exc

    @staticmethod
    def _safe_autocorrelation(values: np.ndarray) -> float:
        if len(values) < 2 or np.std(values[:-1]) == 0 or np.std(values[1:]) == 0:
            return 0.0
        value = float(np.corrcoef(values[:-1], values[1:])[0, 1])
        return value if np.isfinite(value) else 0.0

    @staticmethod
    def _lag_correlation(values: np.ndarray, lag: int) -> float:
        if lag < 1 or len(values) <= lag:
            return 0.0
        return CausalFeatureExtractor._safe_autocorrelation_at_lag(values, lag)

    @staticmethod
    def _safe_autocorrelation_at_lag(values: np.ndarray, lag: int) -> float:
        left = values[:-lag]
        right = values[lag:]
        if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
            return 0.0
        value = float(np.corrcoef(left, right)[0, 1])
        return value if np.isfinite(value) else 0.0

    def _path(self, returns: np.ndarray) -> np.ndarray:
        n_increments = len(returns)
        time = np.arange(n_increments + 1, dtype=float) / n_increments
        cumulative_return = np.concatenate(([0.0], np.cumsum(returns)))
        if self.three_channel_path:
            cumulative_absolute_return = np.concatenate(([0.0], np.cumsum(np.abs(returns))))
            return np.column_stack((time, cumulative_return, cumulative_absolute_return))
        return np.column_stack((time, cumulative_return))

    @staticmethod
    def _shape_statistics(values: np.ndarray) -> Tuple[float, float]:
        if len(values) < 4 or np.std(values) <= np.finfo(float).eps:
            return 0.0, 0.0
        return float(skew(values, bias=False)), float(kurtosis(values, bias=False))

    def _statistical_features(self, returns: np.ndarray) -> np.ndarray:
        if self.statistics_level == "rich":
            return self._rich_statistical_features(returns)
        short = returns[-self.short_window :]
        negative = returns[returns < 0]
        positive = returns[returns > 0]
        cumulative = np.cumsum(returns)
        drawdowns = cumulative - np.maximum.accumulate(cumulative)
        skewness, excess_kurtosis = self._shape_statistics(returns)

        values = np.array(
            [
                np.std(returns) * np.sqrt(252.0),
                np.std(short) * np.sqrt(252.0),
                np.mean(np.abs(returns)),
                np.mean(negative ** 2) if len(negative) else 0.0,
                np.mean(positive ** 2) if len(positive) else 0.0,
                skewness,
                excess_kurtosis,
                self._safe_autocorrelation(returns),
                self._safe_autocorrelation(np.abs(returns)),
                self._safe_autocorrelation(returns ** 2),
                abs(float(np.min(drawdowns))),
            ],
            dtype=float,
        )
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    def _rich_statistical_features(self, returns: np.ndarray) -> np.ndarray:
        """Prespecified B1/B2/B3 hierarchy for fair incremental comparisons."""

        short = returns[-self.short_window :]
        negative = returns[returns < 0]
        positive = returns[returns > 0]
        cumulative = np.cumsum(returns)
        drawdowns = cumulative - np.maximum.accumulate(cumulative)
        quantiles = np.quantile(returns, (0.10, 0.25, 0.50, 0.75, 0.90))
        skewness, excess_kurtosis = self._shape_statistics(returns)
        b1 = np.array(
            [
                np.mean(returns),
                np.std(returns) * np.sqrt(252.0),
                np.std(short) * np.sqrt(252.0),
                np.mean(np.abs(returns)),
                np.mean(negative ** 2) if len(negative) else 0.0,
                np.mean(positive ** 2) if len(positive) else 0.0,
                skewness,
                excess_kurtosis,
                *quantiles,
                abs(float(np.min(drawdowns))),
            ],
            dtype=float,
        )

        dependence = []
        for transform in (lambda x: x, np.abs, lambda x: x ** 2):
            transformed = transform(returns)
            dependence.extend(
                self._lag_correlation(transformed, lag) for lag in (1, 2, 5, 10)
            )
        # Variance ratios capture multi-step persistence without fitting a
        # model or using information outside the current causal window.
        for lag in (2, 5, 10):
            if len(returns) <= lag or np.var(returns) == 0:
                dependence.append(0.0)
            else:
                aggregated = np.convolve(returns, np.ones(lag), mode="valid")
                dependence.append(float(np.var(aggregated) / (lag * np.var(returns))))
        b2 = np.asarray(dependence, dtype=float)

        centered = returns - np.mean(returns)
        spectrum = np.abs(np.fft.rfft(centered)) ** 2
        if len(spectrum):
            spectrum[0] = 0.0
        total_power = float(np.sum(spectrum))
        probabilities = spectrum / total_power if total_power > 0 else np.zeros_like(spectrum)
        positive_probabilities = probabilities[probabilities > 0]
        spectral_entropy = (
            -float(np.sum(positive_probabilities * np.log(positive_probabilities)))
            / np.log(max(len(spectrum), 2))
            if len(positive_probabilities)
            else 0.0
        )
        thirds = np.array_split(spectrum, 3)
        band_power = [
            float(np.sum(band) / total_power) if total_power > 0 else 0.0
            for band in thirds
        ]
        dominant_frequency = (
            float(np.argmax(spectrum) / max(len(returns), 1)) if total_power > 0 else 0.0
        )
        realized_variance = float(np.sum(returns ** 2))
        realized_quarticity = float(len(returns) * np.sum(returns ** 4) / 3.0)
        bipower = (
            float((np.pi / 2.0) * np.sum(np.abs(returns[1:] * returns[:-1])))
            if len(returns) > 1
            else 0.0
        )
        jump_fraction = (
            max(realized_variance - bipower, 0.0) / realized_variance
            if realized_variance > 0
            else 0.0
        )
        sign_changes = float(np.mean(np.signbit(returns[1:]) != np.signbit(returns[:-1])))
        turning_points = (
            float(np.mean(np.diff(returns)[:-1] * np.diff(returns)[1:] < 0))
            if len(returns) > 2
            else 0.0
        )
        b3 = np.asarray(
            [
                spectral_entropy,
                *band_power,
                dominant_frequency,
                realized_variance,
                realized_quarticity,
                bipower,
                jump_fraction,
                sign_changes,
                turning_points,
            ],
            dtype=float,
        )
        return np.nan_to_num(
            np.concatenate((b1, b2, b3)), nan=0.0, posinf=0.0, neginf=0.0
        )

    def _feature_names(self) -> Tuple[str, ...]:
        legacy_statistical_names = (
            "realized_volatility_long",
            "realized_volatility_short",
            "mean_absolute_return",
            "downside_semivariance",
            "upside_semivariance",
            "skewness",
            "kurtosis",
            "return_autocorrelation_1",
            "absolute_return_autocorrelation_1",
            "squared_return_autocorrelation_1",
            "maximum_log_drawdown",
        )
        if self.statistics_level == "legacy":
            statistical_names = legacy_statistical_names
        else:
            statistical_names = (
                "b1_mean_return",
                "b1_realized_volatility_long",
                "b1_realized_volatility_short",
                "b1_mean_absolute_return",
                "b1_downside_semivariance",
                "b1_upside_semivariance",
                "b1_skewness",
                "b1_kurtosis",
                "b1_quantile_10",
                "b1_quantile_25",
                "b1_quantile_50",
                "b1_quantile_75",
                "b1_quantile_90",
                "b1_maximum_log_drawdown",
                *tuple(
                    "b2_{}_autocorrelation_{}".format(transform, lag)
                    for transform in ("return", "absolute_return", "squared_return")
                    for lag in (1, 2, 5, 10)
                ),
                "b2_variance_ratio_2",
                "b2_variance_ratio_5",
                "b2_variance_ratio_10",
                "b3_spectral_entropy",
                "b3_low_band_power",
                "b3_middle_band_power",
                "b3_high_band_power",
                "b3_dominant_frequency",
                "b3_realized_variance",
                "b3_realized_quarticity",
                "b3_bipower_variation",
                "b3_jump_fraction",
                "b3_sign_change_rate",
                "b3_turning_point_rate",
            )
        if self.signature_kind == "log":
            signature_length = iisignature.logsiglength(
                self.path_dimension, self.signature_level
            )
            signature_prefix = "logsignature"
        else:
            signature_length = iisignature.siglength(
                self.path_dimension, self.signature_level
            )
            signature_prefix = "signature"
        signature_names = tuple(
            "{}_{}".format(signature_prefix, index) for index in range(signature_length)
        )
        if self.representation == "statistics":
            return statistical_names
        if self.representation == "signature":
            return signature_names
        return statistical_names + signature_names

    def _ordered_returns(self, returns: np.ndarray, seed: int) -> np.ndarray:
        if self.increment_order == "original":
            return returns
        rng = np.random.default_rng(seed)
        if self.increment_order == "shuffle":
            return returns[rng.permutation(len(returns))]
        blocks = [returns[start : start + self.block_size] for start in range(0, len(returns), self.block_size)]
        return np.concatenate([blocks[index] for index in rng.permutation(len(blocks))])

    def _features(self, returns: np.ndarray, order_seed: int = 0) -> np.ndarray:
        statistics = self._statistical_features(returns)
        if self.representation == "statistics":
            return statistics
        path = self._path(self._ordered_returns(returns, order_seed))
        if self.signature_kind == "log":
            signature_features = np.asarray(
                iisignature.logsig(path, self._logsignature_spec), dtype=float
            )
        else:
            signature_features = np.asarray(
                iisignature.sig(path, self.signature_level), dtype=float
            )
        if self.representation == "signature":
            return signature_features
        return np.concatenate((statistics, signature_features))

    def extract_path(self, path: SimulatedPath) -> WindowDataset:
        frame = path.frame
        required_columns = {"LogReturn", "Regime"}
        missing = required_columns - set(frame.columns)
        if missing:
            raise ValueError("Missing columns: {}".format(sorted(missing)))
        if len(frame) < self.window_size:
            raise ValueError("Path is shorter than window_size.")

        returns = frame["LogReturn"].to_numpy(dtype=float)
        regimes = frame["Regime"].to_numpy(dtype=int)
        rows = []
        labels = []
        end_times = []

        # end is exclusive.  The prediction at end-1 uses observations no later
        # than end-1, and its target is exactly the state at that endpoint.
        for end in range(self.window_size, len(frame) + 1, self.stride):
            start = end - self.window_size
            stable_path_id = zlib.crc32(path.path_id.encode("utf-8"))
            order_seed = int(
                np.random.SeedSequence(
                    [self.order_seed, path.seed, stable_path_id, end]
                ).generate_state(1, dtype=np.uint32)[0]
            )
            rows.append(self._features(returns[start:end], order_seed=order_seed))
            labels.append(regimes[end - 1])
            end_times.append(end - 1)

        count = len(labels)
        return WindowDataset(
            X=np.asarray(rows, dtype=float),
            y=np.asarray(labels, dtype=int),
            path_ids=np.repeat(path.path_id, count),
            end_times=np.asarray(end_times, dtype=int),
            feature_names=self._feature_names(),
        )


def combine_window_datasets(datasets: Iterable[WindowDataset]) -> WindowDataset:
    datasets = tuple(datasets)
    if not datasets:
        raise ValueError("At least one dataset is required.")
    feature_names = datasets[0].feature_names
    if any(dataset.feature_names != feature_names for dataset in datasets):
        raise ValueError("All datasets must use the same features.")
    return WindowDataset(
        X=np.concatenate([dataset.X for dataset in datasets], axis=0),
        y=np.concatenate([dataset.y for dataset in datasets], axis=0),
        path_ids=np.concatenate([dataset.path_ids for dataset in datasets], axis=0),
        end_times=np.concatenate([dataset.end_times for dataset in datasets], axis=0),
        feature_names=feature_names,
    )


def build_path_level_splits(
    extractor: CausalFeatureExtractor,
    parameter_ranges: ParameterRanges = DEFAULT_LEVEL_SHIFT_RANGES,
    parameter_sampler: Optional[ParameterSampler] = None,
    n_paths: Tuple[int, int, int] = DEFAULT_N_PATHS,
    n_steps: int = 5000,
    master_seed: int = 20260825,
    p_switch: float = 0.002,
    min_dwell: Optional[int] = None,
    stationary_initial_variance: bool = False,
    simulation_substeps: int = 1,
    hurst: Optional[float] = None,
) -> PathLevelSplits:
    """Simulate and split independent paths before extracting any windows.

    ``hurst=None`` uses the standard Heston simulator.  Any other value uses
    ``FBMStochasticVolatilitySimulator`` with that Hurst index as the default
    for every regime; per-regime ``hurst`` entries in the sampled parameters
    override it.
    """

    if len(n_paths) != 3 or any(count < 1 for count in n_paths):
        raise ValueError("n_paths must contain positive train/validation/test counts.")
    if n_steps < extractor.window_size:
        raise ValueError("n_steps must be at least extractor.window_size.")
    if min_dwell is None:
        min_dwell = extractor.window_size

    master_rng = np.random.default_rng(master_seed)
    if hurst is None:
        simulator = HestonPathSimulator(substeps=simulation_substeps)
    else:
        simulator = FBMStochasticVolatilitySimulator(
            hurst=hurst, substeps=simulation_substeps
        )
    split_names = ("train", "validation", "test")
    paths_by_split: Dict[str, Tuple[SimulatedPath, ...]] = {}
    data_by_split: Dict[str, WindowDataset] = {}

    for split_name, count in zip(split_names, n_paths):
        split_paths = []
        for index in range(count):
            path_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
            parameter_seed = int(
                master_rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32)
            )
            parameter_rng = np.random.default_rng(parameter_seed)
            if parameter_sampler is None:
                parameters = sample_regime_parameters(parameter_ranges, parameter_rng)
            else:
                parameters = parameter_sampler(parameter_rng)
            split_paths.append(
                simulator.simulate(
                    path_id="{}_path_{:03d}".format(split_name, index),
                    regime_parameters=parameters,
                    n_steps=n_steps,
                    seed=path_seed,
                    p_switch=p_switch,
                    min_dwell=min_dwell,
                    # Balance the starting state within every partition instead
                    # of making all paths begin in regime zero. This guarantees
                    # coverage when a split contains at least one path per state.
                    initial_regime=index % len(parameters),
                    v0=None if stationary_initial_variance else 0.04,
                )
            )
        paths_by_split[split_name] = tuple(split_paths)
        data_by_split[split_name] = combine_window_datasets(
            extractor.extract_path(path) for path in split_paths
        )

    result = PathLevelSplits(
        train=data_by_split["train"],
        validation=data_by_split["validation"],
        test=data_by_split["test"],
        paths=paths_by_split,
    )
    result.assert_disjoint()
    return result


def parameter_audit_table(splits: PathLevelSplits) -> pd.DataFrame:
    """Return the per-path parameter draws for reproducibility and inspection."""

    rows = []
    for split_name, paths in splits.paths.items():
        for path in paths:
            for regime, parameters in path.regime_parameters.items():
                rows.append(
                    {
                        "split": split_name,
                        "path_id": path.path_id,
                        "path_seed": path.seed,
                        "regime": regime,
                        **parameters,
                    }
                )
    return pd.DataFrame(rows).sort_values(["split", "path_id", "regime"]).reset_index(drop=True)
