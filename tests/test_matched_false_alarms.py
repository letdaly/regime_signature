import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiment_matched_false_alarms as mfa
from regime_detection import causal_detector


class DetectorEquivalenceTests(unittest.TestCase):
    def test_fast_alarm_path_reproduces_causal_detector_decisions(self):
        rng = np.random.default_rng(0)
        for trial in range(6):
            logits = rng.normal(size=(1500, 3)) + 1.5 * np.repeat(
                np.eye(3)[rng.integers(0, 3, size=15)], 100, axis=0
            )
            probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
            end_times = np.arange(49, 49 + len(probabilities))
            for alpha in mfa.ALPHAS:
                smoothed = mfa.smooth_probabilities(probabilities, alpha)
                proposed = np.argmax(smoothed, axis=1)
                confidence = smoothed[np.arange(len(proposed)), proposed]
                for confirmations in mfa.CONFIRMATIONS:
                    for threshold in (0.4, 0.6, 0.9):
                        alarms = mfa.detector_alarms(
                            proposed.tolist(), confidence.tolist(), end_times.tolist(), threshold, confirmations
                        )
                        decisions = mfa.decisions_from_alarms(int(proposed[0]), alarms, end_times)
                        expected = causal_detector(probabilities, alpha, threshold, confirmations)
                        np.testing.assert_array_equal(decisions, expected)

    def test_smoothing_starts_at_the_first_probability_vector(self):
        probabilities = np.array([[0.2, 0.7, 0.1], [0.6, 0.3, 0.1], [0.5, 0.4, 0.1]])
        smoothed = mfa.smooth_probabilities(probabilities, 0.5)
        np.testing.assert_allclose(smoothed[0], probabilities[0])
        np.testing.assert_allclose(smoothed[1], 0.5 * probabilities[1] + 0.5 * probabilities[0])


class MatchingTests(unittest.TestCase):
    def test_state_agnostic_matching_counts_wrong_state_alarms_as_detections(self):
        true_events = [(100, 1), (400, 2)]
        alarms = [(120, 2), (130, 1), (600, 0)]
        delays, state_correct, false = mfa.match_alarms(true_events, alarms, tolerance=100)
        self.assertEqual(delays, [20])
        self.assertEqual(state_correct, [False])
        self.assertEqual(false, 2)

    def test_tally_metrics(self):
        tally = mfa.EventTally()
        tally.add([(10, 1), (300, 2)], [(30, 1), (350, 0), (700, 1)], 1000, 100)
        metrics = tally.metrics(state_aware=True, tolerance=100)
        self.assertEqual(metrics["detection_probability"], 1.0)
        self.assertEqual(metrics["state_detection_probability"], 0.5)
        self.assertEqual(metrics["false_alarms_per_1000"], 1.0)
        self.assertEqual(metrics["mean_delay"], 35.0)


class CusumTests(unittest.TestCase):
    def test_cusum_finds_a_variance_shift_and_is_quiet_without_one(self):
        rng = np.random.default_rng(1)
        returns = np.concatenate((rng.normal(0, 0.01, 800), rng.normal(0, 0.03, 800)))
        x = mfa.log_squared_returns(returns).tolist()
        alarms = mfa.cusum_alarms(x, 49, allowance=0.5, half_life=250, limit=30.0)
        self.assertTrue(alarms, "no alarm on a threefold volatility shift")
        first_time, direction = alarms[0]
        self.assertGreaterEqual(first_time, 800)
        self.assertLess(first_time, 900)
        self.assertEqual(direction, 1)
        quiet = mfa.cusum_alarms(mfa.log_squared_returns(rng.normal(0, 0.01, 5000)).tolist(), 49, 0.5, 250, 60.0)
        self.assertLessEqual(len(quiet), 1)


class SelectionTests(unittest.TestCase):
    def test_selection_prefers_highest_detection_within_budget(self):
        sweep = pd.DataFrame(
            {
                "threshold": [0.5, 0.6, 0.7, 0.9],
                "false_alarms_per_1000": [3.0, 1.0, 0.9, 0.2],
                "detection_probability": [0.9, 0.8, 0.82, 0.3],
                "mean_delay": [10, 20, 25, 40],
            }
        )
        chosen, feasible = mfa.select_setting(sweep, 1.0)
        self.assertTrue(feasible)
        self.assertEqual(chosen["threshold"], 0.7)
        chosen, feasible = mfa.select_setting(sweep, 0.1)
        self.assertFalse(feasible)
        self.assertEqual(chosen["threshold"], 0.9)


if __name__ == "__main__":
    unittest.main()
