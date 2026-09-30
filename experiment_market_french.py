"""Market evidence: next-month volatility regimes.

Daily value-weighted market returns for the United States and for developed
markets excluding the United States (Kenneth French Data Library) are turned
into the same causal windows and feature blocks as the simulation grid, and a
walk-forward forecast of the next-month realized-volatility tercile is scored
out of sample.

Protocol, fixed before any out-of-sample number is inspected.  Run

    python experiment_market_french.py --market both --download --freeze

first: it fetches the two archives, writes ``config.json`` with the frozen
sample definition and the SHA-256 of every raw file, and exits without fitting
anything.  Commit that file, then run the same command without ``--freeze``.

* Sample 1990-07-01 to 2025-12-31 (the developed-ex-US daily file starts in
  July 1990); log returns are formed from the market excess return plus the
  risk-free rate.
* Target: the tercile of annualized realized volatility over the next 21
  trading days, with cut-points estimated on training data only.
* Features: the 50-return windows and blocks of the simulation grid
  (Statistics, B3, level-3 raw and log signatures on the origin-anchored
  three-channel path), with Statistics extended by HAR-type averages of
  squared returns over 1, 5 and 22 days.  The HAR columns carry the ``b1_``
  prefix, so B3 remains a strict superset of Statistics here too.
* Models are refitted each year on an expanding window with a 21-day gap
  before the test year, so no training label overlaps the test period.
  Hyperparameters and each representation's learner are selected on the last
  three years of the training window; the same embargo separates that
  validation block from the fitting block.
* Accuracy differences: stationary bootstrap, mean block length 21 days.
* Economics: volatility-managed exposure w_t = c / sigma_hat_t^2 capped at
  two, with the Ledoit-Wolf (2008) test of equal Sharpe ratios.

Outputs, under ``results/market_french/<market>/``: ``config.json``,
``folds/fold_<year>.npz`` (one per test year, so an interrupted run resumes),
``folds.csv``, ``cells.csv``, ``summary.csv`` (accuracy table) and ``economics.csv``
(volatility-managed exposure table).
"""

import argparse
import hashlib
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone

from experiment_unified_grid import (
    BLOCK_PREFIXES,
    CELLS,
    HMM_RETURN_SCALE,
    HMM_STATES,
    LEARNERS,
    STATISTICS_COLUMNS,
    cell_columns,
    fit_cell,
    fit_hmm,
    forward_filter,
    hmm_parameters,
    match_states,
)
from regime_detection import REGIME_IDS, ordered_probabilities
from regime_pipeline import CausalFeatureExtractor, SimulatedPath, WindowDataset


FRENCH_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
MARKETS: Dict[str, Dict[str, str]] = {
    "us": {
        "archive": "F-F_Research_Data_Factors_daily_CSV.zip",
        "label": "United States",
    },
    "developed_ex_us": {
        "archive": "Developed_ex_US_3_Factors_Daily_CSV.zip",
        "label": "Developed ex US",
    },
}

# Frozen sample definition (see the module docstring).  --freeze pins the raw
# files these dates were read from; nothing downstream may widen them.
SAMPLE_START = "1990-07-01"
SAMPLE_END = "2025-12-31"
FIRST_TEST_YEAR = 2005
LAST_TEST_YEAR = 2025

WINDOW = 50
HORIZON = 21  # trading days of forward realized volatility, and the embargo
VALIDATION_YEARS = 3
HAR_LAGS = (1, 5, 22)
TRADING_DAYS = 252.0
N_TERCILES = len(REGIME_IDS)

# Table-1 rows of the market section; a subset of the simulation grid's cells.
MARKET_CELLS = (
    "hmm",
    "statistics",
    "statistics_logsig",
    "statistics_rawsig",
    "b3",
    "b3_logsig",
    "b3_rawsig",
)
BASE_OF = {
    "statistics_logsig": "statistics",
    "statistics_rawsig": "statistics",
    "b3_logsig": "b3",
    "b3_rawsig": "b3",
}
# Exposure rules of the economic table, in printing order (buy-and-hold is added there).
ECONOMIC_CELLS = ("hmm", "statistics", "statistics_logsig", "b3_rawsig")
SHARPE_BENCHMARK_CELL = "statistics"

BOOTSTRAP_BLOCK = 21
BOOTSTRAP_RESAMPLES = 2000
SHARPE_RESAMPLES = 1000
LEVERAGE_CAP = 2.0
BASE_SEED = 20260922


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def resolve_source(market: str, data_dir: Path, download: bool) -> Path:
    """Locate one market's raw file, downloading the French archive if allowed.

    A ``<market>.csv`` in ``data_dir`` wins over the archive, so a frozen
    extract (or a synthetic file in a smoke test) can replace the download.
    """

    data_dir.mkdir(parents=True, exist_ok=True)
    extracted = data_dir / "{}.csv".format(market)
    if extracted.exists():
        return extracted
    archive = data_dir / MARKETS[market]["archive"]
    if not archive.exists():
        if not download:
            raise FileNotFoundError(
                "{} is missing and downloading is disabled. Place the French "
                "archive there or pass --download.".format(archive)
            )
        url = FRENCH_BASE_URL + MARKETS[market]["archive"]
        with urllib.request.urlopen(url, timeout=180) as response:
            archive.write_bytes(response.read())
    return archive


def read_french_daily(source: Path) -> pd.DataFrame:
    """Parse one daily factor file into decimal ``MktRF`` and ``RF`` columns."""

    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not names:
                raise ValueError("{} holds no CSV member.".format(source))
            text = archive.read(names[0]).decode("latin-1")
    else:
        text = source.read_text(encoding="latin-1")

    lines = text.splitlines()
    headers = [index for index, line in enumerate(lines) if "Mkt-RF" in line]
    if not headers:
        raise ValueError("{} has no 'Mkt-RF' header line.".format(source))
    columns = [part.strip() for part in lines[headers[0]].split(",")][1:]
    dates: List[str] = []
    values: List[List[float]] = []
    for line in lines[headers[0] + 1 :]:
        parts = [part.strip() for part in line.split(",")]
        # The daily files end with a copyright line; annual blocks (4-digit
        # keys) never appear in them, so an 8-digit key is the only data row.
        if len(parts) != len(columns) + 1 or not (len(parts[0]) == 8 and parts[0].isdigit()):
            continue
        dates.append(parts[0])
        values.append([float(part) for part in parts[1:]])
    if not dates:
        raise ValueError("{} holds no daily observations.".format(source))

    frame = pd.DataFrame(values, columns=columns, index=pd.to_datetime(dates, format="%Y%m%d"))
    frame = frame.sort_index()
    missing = {"Mkt-RF", "RF"} - set(frame.columns)
    if missing:
        raise ValueError("{} lacks the columns {}.".format(source, sorted(missing)))
    # French reports percent; -99.99 and -999 are the library's missing codes.
    frame = frame[["Mkt-RF", "RF"]].rename(columns={"Mkt-RF": "MktRF"}) / 100.0
    frame = frame[(frame["MktRF"] > -0.99) & (frame["RF"] > -0.99)]
    return frame


def load_market(
    market: str, data_dir: Path, sample_start: str, sample_end: str, download: bool
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Return the frozen sample of one market and the provenance of its file."""

    source = resolve_source(market, data_dir, download)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    raw = read_french_daily(source)
    frame = raw.loc[str(sample_start) : str(sample_end)].copy()
    if frame.empty:
        raise ValueError("{} has no observations inside the frozen sample.".format(market))
    frame["LogReturn"] = np.log1p(frame["MktRF"] + frame["RF"])
    frame["ExcessReturn"] = frame["MktRF"]
    provenance = {
        "market": market,
        "label": MARKETS[market]["label"],
        "source": str(source),
        "sha256": digest,
        "raw_first_date": str(raw.index[0].date()),
        "raw_last_date": str(raw.index[-1].date()),
        "sample_first_date": str(frame.index[0].date()),
        "sample_last_date": str(frame.index[-1].date()),
        "observations": int(len(frame)),
    }
    return frame, provenance


# ---------------------------------------------------------------------------
# Windows, features and targets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketDataset:
    """One market's causal windows, their features and their forward target."""

    market: str
    dates: pd.DatetimeIndex  # (n,) trading days of the frozen sample
    log_returns: np.ndarray  # (n,)
    excess_returns: np.ndarray  # (n,) simple excess returns
    X: np.ndarray  # (m, p) features of the windows ending at end_index
    feature_names: Tuple[str, ...]
    end_index: np.ndarray  # (m,) index of each window's endpoint in dates
    forward_rv: np.ndarray  # (m,) annualized RV over the next HORIZON days

    @property
    def window_dates(self) -> pd.DatetimeIndex:
        return self.dates[self.end_index]

    def labelled(self) -> np.ndarray:
        """Windows whose forward target and next-day return both exist."""

        return np.isfinite(self.forward_rv) & (self.end_index + 1 < len(self.dates))


def _market_path(market: str, returns: np.ndarray, seed: int = 0) -> SimulatedPath:
    """Wrap a return series as a SimulatedPath so the extractors can read it."""

    frame = pd.DataFrame({"LogReturn": returns, "Regime": np.zeros(len(returns), dtype=int)})
    return SimulatedPath(path_id=market, frame=frame, regime_parameters={}, seed=seed)


def har_features(returns: np.ndarray, end_index: np.ndarray) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Backward-looking HAR averages of squared returns at each window endpoint."""

    squared = returns ** 2
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    columns = []
    for lag in HAR_LAGS:
        totals = cumulative[end_index + 1] - cumulative[end_index + 1 - lag]
        columns.append(totals / float(lag))
    names = tuple("b1_har_squared_return_{}".format(lag) for lag in HAR_LAGS)
    return np.column_stack(columns), names


def forward_realized_volatility(returns: np.ndarray, end_index: np.ndarray, horizon: int) -> np.ndarray:
    """Annualized realized volatility over the ``horizon`` days after each endpoint."""

    squared = returns ** 2
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    values = np.full(len(end_index), np.nan)
    complete = end_index + horizon < len(returns)
    ends = end_index[complete] + 1 + horizon
    starts = end_index[complete] + 1
    values[complete] = np.sqrt(TRADING_DAYS * (cumulative[ends] - cumulative[starts]) / horizon)
    return values


def build_market_dataset(
    market: str, frame: pd.DataFrame, window: int, horizon: int
) -> MarketDataset:
    """Extract every block of the simulation grid on this market's windows."""

    returns = frame["LogReturn"].to_numpy(dtype=float)
    path = _market_path(market, returns)
    base = CausalFeatureExtractor(
        window_size=window,
        short_window=min(10, window),
        signature_level=3,
        representation="combined",
        signature_kind="raw",
        statistics_level="rich",
    ).extract_path(path)
    logsig = CausalFeatureExtractor(
        window_size=window,
        short_window=min(10, window),
        signature_level=3,
        representation="signature",
        signature_kind="log",
    ).extract_path(path)
    if not np.array_equal(base.end_times, logsig.end_times):
        raise AssertionError("Window alignment differs between extractors.")

    end_index = np.asarray(base.end_times, dtype=int)
    har, har_names = har_features(returns, end_index)
    return MarketDataset(
        market=market,
        dates=pd.DatetimeIndex(frame.index),
        log_returns=returns,
        excess_returns=frame["ExcessReturn"].to_numpy(dtype=float),
        X=np.concatenate((base.X, logsig.X, har), axis=1),
        feature_names=base.feature_names + logsig.feature_names + har_names,
        end_index=end_index,
        forward_rv=forward_realized_volatility(returns, end_index, horizon),
    )


def market_block_columns(feature_names: Sequence[str]) -> Dict[str, np.ndarray]:
    """Block column indices, with the HAR columns inside Statistics and B3."""

    names = tuple(feature_names)
    position = {name: index for index, name in enumerate(names)}
    statistics = tuple(STATISTICS_COLUMNS) + tuple(
        "b1_har_squared_return_{}".format(lag) for lag in HAR_LAGS
    )
    missing = [name for name in statistics if name not in position]
    if missing:
        raise ValueError("Feature matrix lacks Statistics columns: {}".format(missing))
    selections = {"statistics": np.asarray([position[name] for name in statistics], dtype=int)}
    for block, prefixes in BLOCK_PREFIXES.items():
        columns = np.asarray(
            [index for index, name in enumerate(names) if name.startswith(prefixes)], dtype=int
        )
        if len(columns):
            selections[block] = columns
    return selections


# ---------------------------------------------------------------------------
# Walk-forward folds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fold:
    """One test year with its training window, validation block and labels."""

    year: int
    train: np.ndarray  # boolean masks over the windows of a MarketDataset
    validation: np.ndarray
    test: np.ndarray
    cut_points: np.ndarray  # tercile cut-points, estimated on training data only
    labels: np.ndarray  # (m,) labels under those cut-points; -1 where undefined

    @property
    def pool(self) -> np.ndarray:
        """Everything the fold may fit on: the training block and the validation block.

        The embargoed windows between the two belong to neither, so they enter
        no fit, no cut-point and no exposure scale.
        """

        return self.train | self.validation


def build_folds(
    data: MarketDataset,
    first_year: int,
    last_year: int,
    horizon: int,
    validation_years: int,
    minimum_train: int,
) -> List[Fold]:
    """Expanding-window folds with a ``horizon``-day embargo before each block."""

    labelled = data.labelled()
    window_dates = data.window_dates
    years = np.asarray(window_dates.year)
    folds: List[Fold] = []
    for year in range(first_year, last_year + 1):
        test = labelled & (years == year)
        if not test.any():
            continue
        # A training label spans t+1 .. t+horizon, so t + horizon < test start
        # keeps every training label strictly inside the training period.
        test_start = int(data.end_index[test].min())
        pool = labelled & (data.end_index + horizon < test_start)
        validation = pool & (years >= year - validation_years)
        if not validation.any():
            continue
        validation_start = int(data.end_index[validation].min())
        train = pool & (data.end_index + horizon < validation_start)
        if int(train.sum()) < minimum_train:
            continue
        fitted = train | validation
        cut_points = np.quantile(data.forward_rv[fitted], (1.0 / 3.0, 2.0 / 3.0))
        labels = np.full(len(data.end_index), -1, dtype=int)
        labels[labelled] = np.digitize(data.forward_rv[labelled], cut_points)
        folds.append(
            Fold(
                year=year,
                train=train,
                validation=validation,
                test=test,
                cut_points=np.asarray(cut_points, dtype=float),
                labels=labels,
            )
        )
    if not folds:
        raise ValueError("No test year has enough training history.")
    return folds


# ---------------------------------------------------------------------------
# Markov-switching benchmark
# ---------------------------------------------------------------------------


def hmm_probabilities(
    data: MarketDataset, fold: Fold, seed: int
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Forward-filtered state probabilities at every window endpoint.

    The Gaussian HMM is refitted on this fold's training returns only and its
    states are matched to labels on the training window; the filter itself
    runs over the whole sample, which is causal because the recursion at t
    uses returns up to t.
    """

    pool = fold.pool
    train_end = int(data.end_index[pool].max()) + 1
    model, diagnostics = fit_hmm([_market_path(data.market, data.log_returns[:train_end], seed)], seed)
    start, transition, means, variances = hmm_parameters(model)
    filtered = forward_filter(
        start, transition, means, variances, data.log_returns * HMM_RETURN_SCALE
    )[data.end_index]
    if len(filtered) != len(data.end_index):
        raise AssertionError("HMM filter rows do not align with windows.")
    order = match_states(np.argmax(filtered[pool], axis=1), fold.labels[pool])
    matched = filtered[:, order]
    for label in REGIME_IDS:
        diagnostics["hmm_state_std_{}".format(label)] = float(np.sqrt(variances[order[label]]))
        diagnostics["hmm_persistence_{}".format(label)] = float(
            transition[order[label], order[label]]
        )
    diagnostics["hmm_direct_test_ba"] = balanced_accuracy(
        fold.labels[fold.test], np.argmax(matched[fold.test], axis=1)
    )
    return matched, diagnostics


# ---------------------------------------------------------------------------
# One fold
# ---------------------------------------------------------------------------


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unweighted mean of the recalls of the labels that occur in ``y_true``."""

    recalls = [
        float(np.mean(y_pred[y_true == label] == label))
        for label in REGIME_IDS
        if np.any(y_true == label)
    ]
    return float(np.mean(recalls)) if recalls else float("nan")


def _dataset(X: np.ndarray, y: np.ndarray, mask: np.ndarray, names: Tuple[str, ...]) -> WindowDataset:
    rows = np.flatnonzero(mask)
    return WindowDataset(
        X=X[rows],
        y=y[rows],
        path_ids=np.repeat("market", len(rows)),
        end_times=rows,
        feature_names=names,
    )


def fit_fold(
    data: MarketDataset,
    fold: Fold,
    cells: Sequence[str],
    learners: Sequence[str],
    trees: int,
    model_jobs: int,
    seed: int,
    refit_on_full_window: bool,
) -> Tuple[List[Dict[str, object]], Dict[str, np.ndarray], Dict[str, object]]:
    """Fit every cell on one fold; return its rows, test probabilities and metadata."""

    filtered, diagnostics = hmm_probabilities(data, fold, seed)
    names = data.feature_names + tuple("hmm_{}".format(label) for label in REGIME_IDS)
    X = np.concatenate((data.X, filtered), axis=1)
    columns = cell_columns(market_block_columns(names), cells)

    pool = fold.pool
    train = _dataset(X, fold.labels, fold.train, names)
    validation = _dataset(X, fold.labels, fold.validation, names)
    full = _dataset(X, fold.labels, pool, names)
    test_rows = np.flatnonzero(fold.test)
    y_test = fold.labels[test_rows]

    rows: List[Dict[str, object]] = []
    probabilities: Dict[str, np.ndarray] = {}
    selected: Dict[str, Dict[str, object]] = {}
    for cell in cells:
        best = None
        for learner in learners:
            model, label, _value, validation_ba, _probabilities = fit_cell(
                learner, train, validation, columns[cell], seed, trees, model_jobs
            )
            rows.append(
                {
                    "year": fold.year,
                    "cell": cell,
                    "learner": learner,
                    "setting": label,
                    "n_features": int(len(columns[cell])),
                    "validation_ba": float(validation_ba),
                    "n_train": int(len(train)),
                    "n_validation": int(len(validation)),
                    "n_test": int(len(test_rows)),
                }
            )
            if best is None or validation_ba > best[2]:
                best = (model, learner, float(validation_ba), label)
        model, learner, validation_ba, setting = best
        # The selected setting is refitted on the whole training window so the
        # most recent three years are not discarded before the test year.
        if refit_on_full_window:
            model = clone(model).fit(full.X[:, columns[cell]], full.y)
        test_probabilities = ordered_probabilities(model, X[np.ix_(test_rows, columns[cell])])
        pool_probabilities = ordered_probabilities(model, full.X[:, columns[cell]])
        if not (np.isfinite(test_probabilities).all() and np.isfinite(pool_probabilities).all()):
            raise FloatingPointError("{} returned non-finite probabilities.".format(cell))
        probabilities[cell] = test_probabilities.astype(np.float32)
        test_ba = balanced_accuracy(y_test, np.argmax(test_probabilities, axis=1))
        for row in rows:
            if row["year"] == fold.year and row["cell"] == cell and row["learner"] == learner:
                row["selected"] = True
                row["test_ba"] = test_ba
        selected[cell] = {
            "learner": learner,
            "setting": setting,
            "validation_ba": validation_ba,
            "test_ba": test_ba,
            # Kept for the exposure rule: in-sample training-window fit is used
            # only to set the scale constant c, never to score accuracy.
            "pool_probabilities": pool_probabilities,
        }

    for row in rows:
        row.setdefault("selected", False)
        row.setdefault("test_ba", float("nan"))

    metadata = {
        "year": fold.year,
        "cut_points": fold.cut_points.tolist(),
        "regime_variances": regime_variances(data, fold).tolist(),
        "hmm": {key: float(value) for key, value in diagnostics.items()},
        "selected": {
            cell: {
                key: value
                for key, value in entry.items()
                if key != "pool_probabilities"
            }
            for cell, entry in selected.items()
        },
    }
    exposures = {
        cell: exposure_weights(data, fold, entry["pool_probabilities"], probabilities[cell])
        for cell, entry in selected.items()
    }
    return rows, {"probabilities": probabilities, "exposures": exposures}, metadata


# ---------------------------------------------------------------------------
# Volatility-managed exposure
# ---------------------------------------------------------------------------


def regime_variances(data: MarketDataset, fold: Fold) -> np.ndarray:
    """Training-sample variance implied by each regime label."""

    pool = fold.pool
    pooled = float(np.mean(data.forward_rv[pool] ** 2))
    variances = np.full(N_TERCILES, pooled, dtype=float)
    for label in REGIME_IDS:
        mask = pool & (fold.labels == label)
        if mask.any():
            variances[label] = float(np.mean(data.forward_rv[mask] ** 2))
    return variances


def exposure_weights(
    data: MarketDataset,
    fold: Fold,
    pool_probabilities: np.ndarray,
    test_probabilities: np.ndarray,
    cap: float = LEVERAGE_CAP,
) -> np.ndarray:
    """Next-day exposure ``w_t = c / sigma_hat_t^2``, capped at ``cap``.

    ``c`` equates the strategy's and the market's volatility over the training
    window, so it is the only quantity read off an in-sample fit; the test-period
    weights themselves use out-of-sample probabilities.
    """

    variances = regime_variances(data, fold)
    pool_rows = np.flatnonzero(fold.pool)
    test_rows = np.flatnonzero(fold.test)
    pool_excess = data.excess_returns[data.end_index[pool_rows] + 1]
    # Probability rows sum to one and every regime variance is positive, so the
    # implied variance cannot vanish; the guard is only for the spurious
    # Accelerate matmul warnings noted in experiment_unified_grid.fit_cell.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        raw_pool = 1.0 / (np.asarray(pool_probabilities, dtype=float) @ variances)
        raw_test = 1.0 / (np.asarray(test_probabilities, dtype=float) @ variances)
    if not (np.isfinite(raw_pool).all() and np.isfinite(raw_test).all()):
        raise FloatingPointError("Implied variance vanished in {}.".format(fold.year))
    managed = raw_pool * pool_excess
    scale = float(np.std(pool_excess) / np.std(managed)) if np.std(managed) > 0 else 0.0
    return np.minimum(scale * raw_test, cap)


def performance(returns: np.ndarray, weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Annualized mean, volatility, Sharpe ratio, maximum drawdown and turnover."""

    returns = np.asarray(returns, dtype=float)
    mean = float(np.mean(returns))
    volatility = float(np.std(returns, ddof=1))
    equity = np.cumprod(1.0 + returns)
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    row = {
        "mean_annualized_pct": 100.0 * TRADING_DAYS * mean,
        "volatility_annualized_pct": 100.0 * np.sqrt(TRADING_DAYS) * volatility,
        "sharpe": float(np.sqrt(TRADING_DAYS) * mean / volatility) if volatility > 0 else float("nan"),
        "max_drawdown_pct": 100.0 * float(np.min(drawdown)),
    }
    row["turnover"] = (
        float(np.mean(np.abs(np.diff(np.asarray(weights, dtype=float))))) if weights is not None else float("nan")
    )
    return row


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def stationary_bootstrap_indices(
    n: int, mean_block: float, n_resamples: int, rng: np.random.Generator
) -> Iterator[np.ndarray]:
    """Politis-Romano stationary-bootstrap index vectors of length ``n``."""

    probability = 1.0 / float(mean_block)
    offsets = np.arange(n)
    for _ in range(n_resamples):
        restart = rng.random(n) < probability
        restart[0] = True
        last = np.maximum.accumulate(np.where(restart, offsets, -1))
        starts = rng.integers(0, n, size=n)
        yield (starts[last] + (offsets - last)) % n


def bootstrap_gain(
    y: np.ndarray,
    augmented: np.ndarray,
    base: np.ndarray,
    mean_block: float,
    n_resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """Balanced-accuracy gain with a stationary-bootstrap interval, in points."""

    rng = np.random.default_rng(seed)
    point = balanced_accuracy(y, augmented) - balanced_accuracy(y, base)
    draws = np.asarray(
        [
            balanced_accuracy(y[index], augmented[index]) - balanced_accuracy(y[index], base[index])
            for index in stationary_bootstrap_indices(len(y), mean_block, n_resamples, rng)
        ],
        dtype=float,
    )
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, (tail, 1.0 - tail))
    share_below = float(np.mean(draws - point <= -abs(point)))
    share_above = float(np.mean(draws - point >= abs(point)))
    return {
        "gain_points": 100.0 * point,
        # Reverse-percentile ("basic") interval; the raw percentiles are kept
        # because the two differ visibly when the gain distribution is skewed.
        "ci_low": 100.0 * (2.0 * point - high),
        "ci_high": 100.0 * (2.0 * point - low),
        "percentile_low": 100.0 * low,
        "percentile_high": 100.0 * high,
        "bootstrap_p": float(min(1.0, share_below + share_above)),
        "resamples": int(n_resamples),
        "mean_block": float(mean_block),
    }


def _hac_covariance(values: np.ndarray, bandwidth: float) -> np.ndarray:
    """Bartlett-kernel HAC covariance of the columns of ``values``."""

    centred = values - values.mean(axis=0)
    n = len(centred)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        psi = centred.T @ centred / n
        for lag in range(1, int(bandwidth) + 1):
            weight = 1.0 - lag / (bandwidth + 1.0)
            cross = centred[lag:].T @ centred[:-lag] / n
            psi = psi + weight * (cross + cross.T)
    return psi


def sharpe_difference(a: np.ndarray, b: np.ndarray, bandwidth: float) -> Tuple[float, float]:
    """Daily Sharpe-ratio difference and its HAC standard error (Ledoit-Wolf)."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mu1, mu2 = float(np.mean(a)), float(np.mean(b))
    g1, g2 = float(np.mean(a ** 2)), float(np.mean(b ** 2))
    s1, s2 = np.sqrt(g1 - mu1 ** 2), np.sqrt(g2 - mu2 ** 2)
    if s1 <= 0 or s2 <= 0:
        return float("nan"), float("nan")
    difference = mu1 / s1 - mu2 / s2
    gradient = np.array(
        [g1 / s1 ** 3, -g2 / s2 ** 3, -mu1 / (2.0 * s1 ** 3), mu2 / (2.0 * s2 ** 3)]
    )
    psi = _hac_covariance(np.column_stack((a, b, a ** 2, b ** 2)), bandwidth)
    variance = float(gradient @ psi @ gradient) / len(a)
    if not np.isfinite(variance):
        return difference, float("nan")
    # The Bartlett kernel makes the quadratic form positive semidefinite, so a
    # negative value here is rounding around an exactly zero difference.
    return difference, float(np.sqrt(max(variance, 0.0)))


def _normal_p_value(difference: float, standard_error: float) -> float:
    if standard_error > 0:
        return float(2.0 * (1.0 - norm.cdf(abs(difference / standard_error))))
    if np.isfinite(standard_error) and difference == 0.0:
        return 1.0
    return float("nan")


def sharpe_test(
    a: np.ndarray,
    b: np.ndarray,
    block: int,
    n_resamples: int,
    seed: int,
) -> Dict[str, float]:
    """Studentized circular-block bootstrap test of equal Sharpe ratios."""

    difference, standard_error = sharpe_difference(a, b, block)
    row = {
        "sharpe_difference": float(np.sqrt(TRADING_DAYS) * difference),
        "sharpe_standard_error": float(np.sqrt(TRADING_DAYS) * standard_error),
        "p_hac": _normal_p_value(difference, standard_error),
    }
    if not np.isfinite(standard_error) or standard_error <= 0 or n_resamples <= 0:
        row["p_value"] = row["p_hac"]
        return row
    rng = np.random.default_rng(seed)
    n = len(a)
    n_blocks = int(np.ceil(n / block))
    observed = abs(difference / standard_error)
    exceedances = 0
    valid = 0
    for _ in range(n_resamples):
        starts = rng.integers(0, n, size=n_blocks)
        index = ((starts[:, None] + np.arange(block)[None, :]).ravel()[:n]) % n
        draw, draw_error = sharpe_difference(a[index], b[index], block)
        if np.isfinite(draw_error) and draw_error > 0:
            valid += 1
            if abs((draw - difference) / draw_error) >= observed:
                exceedances += 1
    row["p_value"] = float((exceedances + 1) / (valid + 1)) if valid else row["p_hac"]
    row["sharpe_resamples"] = int(valid)
    return row


# ---------------------------------------------------------------------------
# Fold storage and assembly
# ---------------------------------------------------------------------------


def save_fold(
    fold_dir: Path, data: MarketDataset, fold: Fold, payload: Dict[str, Dict[str, np.ndarray]], metadata: Dict[str, object]
) -> Path:
    """Write one test year's labels, probabilities, weights and metadata."""

    test_rows = np.flatnonzero(fold.test)
    arrays: Dict[str, np.ndarray] = {
        "dates": data.window_dates[test_rows].asi8,
        "labels": fold.labels[test_rows].astype(np.int8),
        "excess_next": data.excess_returns[data.end_index[test_rows] + 1],
        "metadata": np.asarray(json.dumps(metadata)),
    }
    for cell, probabilities in payload["probabilities"].items():
        arrays["prob_{}".format(cell)] = np.asarray(probabilities, dtype=np.float32)
    for cell, weights in payload["exposures"].items():
        arrays["weight_{}".format(cell)] = np.asarray(weights, dtype=np.float32)
    fold_dir.mkdir(parents=True, exist_ok=True)
    target = fold_dir / "fold_{}.npz".format(fold.year)
    np.savez_compressed(target, **arrays)
    return target


def load_folds(fold_dir: Path, cells: Sequence[str]) -> Dict[str, object]:
    """Concatenate every stored test year into one out-of-sample series."""

    files = sorted(fold_dir.glob("fold_*.npz"), key=lambda path: int(path.stem.split("_")[1]))
    if not files:
        raise FileNotFoundError("No fold files in {}.".format(fold_dir))
    dates, labels, excess = [], [], []
    probabilities = {cell: [] for cell in cells}
    weights = {cell: [] for cell in cells}
    metadata = []
    for file in files:
        with np.load(file, allow_pickle=False) as stored:
            dates.append(stored["dates"])
            labels.append(stored["labels"])
            excess.append(stored["excess_next"])
            for cell in cells:
                probabilities[cell].append(stored["prob_{}".format(cell)])
                weights[cell].append(stored["weight_{}".format(cell)])
            metadata.append(json.loads(str(stored["metadata"])))
    return {
        "dates": pd.DatetimeIndex(np.concatenate(dates)),
        "labels": np.concatenate(labels).astype(int),
        "excess_next": np.concatenate(excess),
        "probabilities": {cell: np.concatenate(parts) for cell, parts in probabilities.items()},
        "weights": {cell: np.concatenate(parts) for cell, parts in weights.items()},
        "metadata": metadata,
    }


def fold_table(metadata: Sequence[Dict[str, object]]) -> pd.DataFrame:
    """Per-year, per-cell, per-learner validation and test balanced accuracy."""

    rows = [row for entry in metadata for row in entry["rows"]]
    return pd.DataFrame(rows).sort_values(["year", "cell", "learner"]).reset_index(drop=True)


def accuracy_table(
    pooled: Dict[str, object], cells: Sequence[str], block: float, resamples: int, seed: int
) -> pd.DataFrame:
    """Market accuracy table: pooled out-of-sample BA and paired gains."""

    labels = pooled["labels"]
    predictions = {
        cell: np.argmax(pooled["probabilities"][cell], axis=1) for cell in cells
    }
    rows = []
    for offset, cell in enumerate(cells):
        row: Dict[str, object] = {
            "cell": cell,
            "n_observations": int(len(labels)),
            "ba_pct": 100.0 * balanced_accuracy(labels, predictions[cell]),
            "accuracy_pct": 100.0 * float(np.mean(predictions[cell] == labels)),
            "base": BASE_OF.get(cell, ""),
        }
        if cell in BASE_OF:
            row.update(
                bootstrap_gain(
                    labels,
                    predictions[cell],
                    predictions[BASE_OF[cell]],
                    block,
                    resamples,
                    seed + offset,
                )
            )
        rows.append(row)
    return pd.DataFrame(rows)


def economics_table(
    pooled: Dict[str, object],
    cells: Sequence[str],
    benchmark: str,
    block: int,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    """Economic table: volatility-managed exposure to the market."""

    excess = pooled["excess_next"]
    strategies = {"buy_and_hold": (excess, None)}
    for cell in cells:
        weights = pooled["weights"][cell]
        strategies[cell] = (weights * excess, weights)
    benchmark_returns = strategies[benchmark][0]
    rows = []
    for offset, (name, (returns, weights)) in enumerate(strategies.items()):
        row: Dict[str, object] = {"rule": name}
        row.update(performance(returns, weights))
        if name not in ("buy_and_hold", benchmark):
            row.update(sharpe_test(returns, benchmark_returns, block, resamples, seed + offset))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def guard_output_directory(output_dir: Path, resume: bool) -> None:
    """Frozen result directories are never overwritten (as in the simulation rows)."""

    existing = output_dir / "summary.csv"
    if existing.exists() and not resume:
        raise FileExistsError(
            "{} already holds results. Pass --resume to continue that run or "
            "choose a new --output-dir.".format(output_dir)
        )


def run_configuration(args: argparse.Namespace, provenance: Dict[str, object]) -> Dict[str, object]:
    return {
        "experiment": "market_french",
        "data": provenance,
        "sample_start": str(args.sample_start),
        "sample_end": str(args.sample_end),
        "first_test_year": int(args.first_test_year),
        "last_test_year": int(args.last_test_year),
        "window_size": int(args.window),
        "horizon": int(args.horizon),
        "embargo": int(args.horizon),
        "validation_years": int(args.validation_years),
        "minimum_train_windows": int(args.minimum_train),
        "har_lags": list(HAR_LAGS),
        "signature_level": 3,
        "signature_path": CausalFeatureExtractor.PATH_CONVENTION,
        "cells": {cell: list(CELLS[cell]) for cell in args.cells},
        "learners": list(args.learners),
        "trees": int(args.trees),
        "refit_on_full_window": bool(args.refit_on_full_window),
        "bootstrap_block": float(args.block_length),
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "sharpe_resamples": int(args.sharpe_resamples),
        "leverage_cap": LEVERAGE_CAP,
        "seed": int(args.seed),
    }


def run_market(market: str, args: argparse.Namespace) -> None:
    """Freeze, fit and summarize one market."""

    output_dir = args.output_dir / market
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, provenance = load_market(
        market, args.data_dir, args.sample_start, args.sample_end, args.download
    )
    configuration = run_configuration(args, provenance)
    (output_dir / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    print(
        "[{}] {} observations, {} to {} (sha256 {}...)".format(
            market,
            provenance["observations"],
            provenance["sample_first_date"],
            provenance["sample_last_date"],
            str(provenance["sha256"])[:12],
        ),
        flush=True,
    )
    if args.freeze:
        print("[{}] frozen; commit config.json before running the fits.".format(market), flush=True)
        return

    guard_output_directory(output_dir, args.resume)
    data = build_market_dataset(market, frame, args.window, args.horizon)
    folds = build_folds(
        data,
        args.first_test_year,
        args.last_test_year,
        args.horizon,
        args.validation_years,
        args.minimum_train,
    )
    fold_dir = output_dir / "folds"
    for fold in folds:
        target = fold_dir / "fold_{}.npz".format(fold.year)
        if args.resume and target.exists():
            print("[{}] {} already stored.".format(market, fold.year), flush=True)
            continue
        rows, payload, metadata = fit_fold(
            data,
            fold,
            args.cells,
            args.learners,
            args.trees,
            args.model_jobs,
            args.seed + fold.year,
            args.refit_on_full_window,
        )
        metadata["rows"] = rows
        save_fold(fold_dir, data, fold, payload, metadata)
        selected = metadata["selected"]
        print(
            "[{}] {}: train {}, test {} | {}".format(
                market,
                fold.year,
                int(fold.train.sum()),
                int(fold.test.sum()),
                ", ".join(
                    "{} {:.1f}% ({})".format(cell, 100.0 * selected[cell]["test_ba"], selected[cell]["learner"])
                    for cell in args.cells
                ),
            ),
            flush=True,
        )

    pooled = load_folds(fold_dir, args.cells)
    fold_table(pooled["metadata"]).to_csv(output_dir / "folds.csv", index=False)
    summary = accuracy_table(
        pooled, args.cells, args.block_length, args.bootstrap_resamples, args.seed
    )
    summary.insert(0, "market", market)
    summary.to_csv(output_dir / "summary.csv", index=False)
    counts = (
        fold_table(pooled["metadata"])
        .query("selected")
        .groupby(["cell", "learner"])
        .size()
        .rename("years_selected")
        .reset_index()
    )
    counts.to_csv(output_dir / "cells.csv", index=False)
    print(summary.to_string(index=False), flush=True)

    if not args.skip_economics:
        economic_cells = [cell for cell in ECONOMIC_CELLS if cell in args.cells]
        benchmark = SHARPE_BENCHMARK_CELL if SHARPE_BENCHMARK_CELL in economic_cells else economic_cells[0]
        economics = economics_table(
            pooled, economic_cells, benchmark, int(args.block_length), args.sharpe_resamples, args.seed
        )
        economics.insert(0, "market", market)
        economics.to_csv(output_dir / "economics.csv", index=False)
        print(economics.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--market", choices=tuple(MARKETS) + ("both",), default="both")
    parser.add_argument("--data-dir", type=Path, default=Path("data/french"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/market_french"))
    parser.add_argument("--sample-start", default=SAMPLE_START)
    parser.add_argument("--sample-end", default=SAMPLE_END)
    parser.add_argument("--first-test-year", type=int, default=FIRST_TEST_YEAR)
    parser.add_argument("--last-test-year", type=int, default=LAST_TEST_YEAR)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--validation-years", type=int, default=VALIDATION_YEARS)
    parser.add_argument("--minimum-train", type=int, default=1000)
    parser.add_argument("--cells", nargs="+", choices=MARKET_CELLS, default=list(MARKET_CELLS))
    parser.add_argument("--learners", nargs="+", choices=LEARNERS, default=list(LEARNERS))
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--model-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--block-length", type=float, default=float(BOOTSTRAP_BLOCK))
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--sharpe-resamples", type=int, default=SHARPE_RESAMPLES)
    parser.add_argument(
        "--no-refit-on-full-window",
        dest="refit_on_full_window",
        action="store_false",
        help="Score the model fitted before the validation block instead of refitting on it.",
    )
    parser.add_argument("--download", action="store_true", help="Fetch missing French archives.")
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Write config.json with the sample definition and file hashes, then stop.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-economics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    markets = tuple(MARKETS) if args.market == "both" else (args.market,)
    for market in markets:
        run_market(market, args)


if __name__ == "__main__":
    main()
