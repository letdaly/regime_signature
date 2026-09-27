import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiment_market_french as market


def _french_text(n_days: int = 400, seed: int = 3) -> str:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("1990-07-02", periods=n_days)
    lines = ["Synthetic daily factors", "", ",Mkt-RF,SMB,HML,RF"]
    for stamp in dates:
        lines.append(
            "{},{:8.2f},{:8.2f},{:8.2f},{:8.5f}".format(
                stamp.strftime("%Y%m%d"), 100 * rng.normal(0, 0.01), 0.0, 0.0, 0.015
            )
        )
    lines.append("Copyright 2026 Kenneth R. French")
    return "\n".join(lines) + "\n"


def _dataset(n_days: int = 900, seed: int = 5) -> market.MarketDataset:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("1990-07-02", periods=n_days)
    returns = rng.normal(0.0, 0.01, n_days)
    frame = pd.DataFrame(
        {"LogReturn": returns, "ExcessReturn": returns - 0.0001}, index=dates
    )
    return market.build_market_dataset("test", frame, window=50, horizon=21)


class FileParsingTests(unittest.TestCase):
    def test_header_block_and_copyright_line_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "us.csv"
            source.write_text(_french_text())
            frame = market.read_french_daily(source)
        self.assertEqual(len(frame), 400)
        self.assertEqual(list(frame.columns), ["MktRF", "RF"])
        # French reports percent; the parser stores decimals.
        self.assertLess(frame["MktRF"].abs().max(), 0.2)


class TargetTests(unittest.TestCase):
    def test_forward_volatility_is_strictly_forward_looking(self):
        returns = np.zeros(200)
        returns[100] = 0.05
        end_index = np.arange(49, 200)
        values = market.forward_realized_volatility(returns, end_index, horizon=21)
        # The target of endpoint t spans t+1 .. t+21, so the jump at t = 100
        # reaches endpoints 79..99 and no other.
        reaches = (end_index >= 79) & (end_index <= 99)
        self.assertTrue(np.all(values[reaches] > 0.0))
        seen = reaches | ~np.isfinite(values)
        self.assertTrue(np.all(values[~seen] == 0.0))
        # The last 21 endpoints have no complete forward window.
        self.assertTrue(np.all(np.isnan(values[end_index > 200 - 1 - 21])))
        self.assertFalse(np.isnan(values[end_index == 200 - 1 - 21][0]))

    def test_har_features_are_backward_looking_averages(self):
        returns = np.arange(1.0, 61.0)
        end_index = np.asarray([59])
        values, names = market.har_features(returns, end_index)
        self.assertEqual(names[0], "b1_har_squared_return_1")
        self.assertAlmostEqual(values[0, 0], 60.0 ** 2)
        self.assertAlmostEqual(values[0, 1], np.mean(np.arange(56.0, 61.0) ** 2))


class FoldTests(unittest.TestCase):
    def test_no_training_label_reaches_the_test_period(self):
        data = _dataset()
        folds = market.build_folds(
            data,
            first_year=1993,
            last_year=1993,
            horizon=21,
            validation_years=1,
            minimum_train=50,
        )
        fold = folds[0]
        test_start = int(data.end_index[fold.test].min())
        pool = fold.pool
        self.assertLess(int(data.end_index[pool].max()) + 21, test_start)
        validation_start = int(data.end_index[fold.validation].min())
        self.assertLess(int(data.end_index[fold.train].max()) + 21, validation_start)
        self.assertFalse(np.any(fold.train & fold.validation))
        self.assertFalse(np.any(pool & fold.test))

    def test_cut_points_come_from_the_training_window_only(self):
        data = _dataset()
        fold = market.build_folds(data, 1993, 1993, 21, 1, 50)[0]
        pool = fold.pool
        expected = np.quantile(data.forward_rv[pool], (1.0 / 3.0, 2.0 / 3.0))
        np.testing.assert_allclose(fold.cut_points, expected)
        shares = np.bincount(fold.labels[pool], minlength=3) / int(pool.sum())
        np.testing.assert_allclose(shares, np.full(3, 1.0 / 3.0), atol=0.01)


class BlockTests(unittest.TestCase):
    def test_statistics_carries_the_har_columns_and_b3_remains_a_superset(self):
        data = _dataset(n_days=200)
        names = data.feature_names + tuple("hmm_{}".format(label) for label in (0, 1, 2))
        selections = market.market_block_columns(names)
        statistics = set(selections["statistics"].tolist())
        self.assertEqual(len(statistics), 14)
        self.assertTrue(statistics <= set(selections["b3"].tolist()))
        self.assertTrue(
            any(names[index].startswith("b1_har_") for index in selections["statistics"])
        )
        self.assertEqual(len(selections["rawsig"]), 39)
        self.assertEqual(len(selections["logsig"]), 14)


class InferenceTests(unittest.TestCase):
    def test_stationary_bootstrap_keeps_blocks_contiguous(self):
        rng = np.random.default_rng(0)
        draws = list(market.stationary_bootstrap_indices(200, 20.0, 50, rng))
        self.assertEqual(len(draws), 50)
        steps = np.concatenate([np.diff(draw) % 200 for draw in draws])
        # A mean block length of 20 leaves about 95% of consecutive positions.
        self.assertGreater(float(np.mean(steps == 1)), 0.85)

    def test_identical_series_give_a_zero_sharpe_difference(self):
        rng = np.random.default_rng(1)
        returns = rng.normal(0.0004, 0.01, 800)
        difference, standard_error = market.sharpe_difference(returns, returns, bandwidth=21)
        self.assertAlmostEqual(difference, 0.0)
        self.assertAlmostEqual(standard_error, 0.0, places=8)

    def test_balanced_accuracy_matches_sklearn(self):
        from sklearn.metrics import balanced_accuracy_score

        rng = np.random.default_rng(2)
        y_true = rng.integers(0, 3, 300)
        y_pred = np.where(rng.random(300) < 0.5, y_true, rng.integers(0, 3, 300))
        self.assertAlmostEqual(
            market.balanced_accuracy(y_true, y_pred), float(balanced_accuracy_score(y_true, y_pred))
        )


class ExposureTests(unittest.TestCase):
    def test_leverage_is_capped_and_variances_follow_the_labels(self):
        data = _dataset()
        fold = market.build_folds(data, 1993, 1993, 21, 1, 50)[0]
        variances = market.regime_variances(data, fold)
        self.assertTrue(np.all(np.diff(variances) > 0))
        pool = int(fold.pool.sum())
        test = int(fold.test.sum())
        certain_high = np.tile(np.array([0.0, 0.0, 1.0]), (pool, 1))
        certain_low = np.tile(np.array([1.0, 0.0, 0.0]), (test, 1))
        weights = market.exposure_weights(data, fold, certain_high, certain_low)
        self.assertTrue(np.all(weights <= market.LEVERAGE_CAP + 1e-12))
        self.assertTrue(np.all(weights > 0))


if __name__ == "__main__":
    unittest.main()
