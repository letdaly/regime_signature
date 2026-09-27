import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import multiplicity_holm as multiplicity


def _reference_holm(p_values):
    """Textbook Holm, written independently of the implementation under test."""

    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    previous = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, max(previous, (n - rank) * p_values[index]))
        adjusted[index] = value
        previous = value
    return adjusted


class HolmTests(unittest.TestCase):
    def test_matches_the_textbook_example(self):
        adjusted = multiplicity.holm([0.01, 0.02, 0.03, 0.04, 0.05])
        np.testing.assert_allclose(adjusted, [0.05, 0.08, 0.09, 0.09, 0.09])

    def test_agrees_with_an_independent_implementation_on_random_inputs(self):
        rng = np.random.default_rng(11)
        for size in (1, 2, 5, 24, 64):
            p_values = rng.random(size) ** 3
            np.testing.assert_allclose(
                multiplicity.holm(p_values), _reference_holm(list(p_values))
            )

    def test_is_monotone_capped_and_never_below_the_raw_value(self):
        rng = np.random.default_rng(3)
        p_values = rng.random(30)
        adjusted = multiplicity.holm(p_values)
        self.assertTrue(np.all(adjusted <= 1.0))
        self.assertTrue(np.all(adjusted >= p_values - 1e-12))
        order = np.argsort(p_values)
        self.assertTrue(np.all(np.diff(adjusted[order]) >= -1e-12))

    def test_a_single_hypothesis_is_left_alone(self):
        np.testing.assert_allclose(multiplicity.holm([0.03]), [0.03])


class PairedTests(unittest.TestCase):
    def test_paired_t_reproduces_the_interval_used_in_the_paper(self):
        from experiment_a_monte_carlo import mean_t_interval

        # Deterministic, so the assertion does not depend on a lucky draw:
        # mean 0.004, sample standard deviation 0.01, 50 replications.
        base = np.linspace(-1.0, 1.0, 50)
        differences = 0.004 + 0.01 * (base - base.mean()) / base.std(ddof=1)
        row = multiplicity.paired_t(differences)
        reference = mean_t_interval(differences)
        self.assertEqual(row["n"], 50)
        self.assertAlmostEqual(row["delta_points"], 100.0 * reference["mean"])
        self.assertAlmostEqual(row["ci_low"], 100.0 * reference["ci_low"])
        self.assertAlmostEqual(row["ci_high"], 100.0 * reference["ci_high"])
        self.assertLess(row["p_raw"], 0.05)

    def test_a_difference_without_variation_is_not_resolved(self):
        for constant in (0.0, 1e-12, 0.5):
            row = multiplicity.paired_t(np.full(50, constant))
            self.assertEqual(row["p_raw"], 1.0)


class TableTests(unittest.TestCase):
    def test_every_reported_contrast_appears_once_per_family(self):
        table = multiplicity.build_table()
        counts = table.groupby("family").size().to_dict()
        self.assertEqual(counts["classification"], 24)
        self.assertEqual(counts["detection"], 32)
        self.assertEqual(counts["market"], 8)
        self.assertEqual(len(table), 64)
        self.assertFalse(table.duplicated(["family", "unit", "learner", "budget", "contrast"]).any())

    def test_corrections_never_resolve_more_than_the_raw_p_values(self):
        table = multiplicity.build_table()
        self.assertTrue((table["p_holm_family"] >= table["p_raw"] - 1e-12).all())
        self.assertTrue((table["p_holm_global"] >= table["p_holm_family"] - 1e-12).all())
        self.assertLessEqual(int(table["resolved_global"].sum()), int(table["resolved_family"].sum()))
        self.assertLessEqual(int(table["resolved_family"].sum()), int(table["resolved_raw"].sum()))


if __name__ == "__main__":
    unittest.main()
