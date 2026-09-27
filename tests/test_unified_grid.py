import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiment_unified_grid as grid
from regime_pipeline import (
    CausalFeatureExtractor,
    HestonPathSimulator,
    build_path_level_splits,
    sample_scale_only_parameters,
)


def _small_path(seed: int = 7, n_steps: int = 120):
    parameters = sample_scale_only_parameters(np.random.default_rng(seed))
    return HestonPathSimulator().simulate(
        "unified_test", parameters, n_steps, seed=seed, p_switch=0.05, min_dwell=10
    )


class BlockAssemblyTests(unittest.TestCase):
    def test_statistics_block_equals_the_legacy_extractor_column_for_column(self):
        path = _small_path()
        legacy = CausalFeatureExtractor(
            window_size=20, short_window=10, representation="statistics"
        ).extract_path(path)
        rich = CausalFeatureExtractor(
            window_size=20,
            short_window=10,
            representation="combined",
            signature_kind="raw",
            statistics_level="rich",
        ).extract_path(path)

        columns = grid.block_columns(rich.feature_names)["statistics"]

        self.assertEqual(len(columns), 11)
        self.assertEqual(
            tuple(rich.feature_names[index] for index in columns), grid.STATISTICS_COLUMNS
        )
        np.testing.assert_allclose(rich.X[:, columns], legacy.X, rtol=0, atol=1e-12)

    def test_grid_splits_hold_every_block_with_shared_shuffle_permutation(self):
        splits = grid.build_grid_splits(
            "scale_only", seed=11, n_paths=(1, 1, 1), n_steps=90, window_size=20, p_switch=0.05
        )
        selections = grid.block_columns(splits.train.feature_names + ("hmm_0", "hmm_1", "hmm_2"))

        self.assertEqual(len(selections["b3"]), 40)
        self.assertEqual(len(selections["rawsig"]), 39)
        self.assertEqual(len(selections["logsig"]), 14)
        self.assertEqual(len(selections["rawsig_shuffled"]), 39)
        self.assertEqual(len(selections["logsig_shuffled"]), 14)
        # Shuffled raw and log coordinates come from the same permuted path:
        # the level-1 raw coordinates are the total increments (permutation
        # invariant), the level-2 Levy-area terms are not.
        raw = splits.train.X[:, selections["rawsig"]]
        shuffled = splits.train.X[:, selections["rawsig_shuffled"]]
        np.testing.assert_allclose(raw[:, :3], shuffled[:, :3], atol=1e-12)
        self.assertGreater(np.max(np.abs(raw[:, 3:12] - shuffled[:, 3:12])), 1e-8)
        cells = grid.cell_columns(selections)
        self.assertEqual(len(cells["b3_rawsig"]), 79)
        self.assertEqual(len(cells["statistics_logsig"]), 25)
        self.assertEqual(len(cells["hmm"]), 3)


class MarkovSwitchingFilterTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.model = GaussianHMM(n_components=3, covariance_type="diag", random_state=0)
        self.model.n_features = 1
        self.model.startprob_ = np.array([0.5, 0.3, 0.2])
        self.model.transmat_ = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
        self.model.means_ = np.array([[0.0], [0.1], [-0.1]])
        self.model.covars_ = np.array([[0.5], [1.0], [2.0]])
        self.observations = rng.normal(size=300)

    def test_forward_filter_is_a_causal_distribution_matching_hmmlearn_at_the_end(self):
        start, transition, means, variances = grid.hmm_parameters(self.model)
        filtered = grid.forward_filter(start, transition, means, variances, self.observations)

        self.assertEqual(filtered.shape, (300, 3))
        np.testing.assert_allclose(filtered.sum(axis=1), 1.0, atol=1e-12)
        # Smoothing equals filtering at the final observation.
        smoothed = self.model.predict_proba(self.observations[:, None])
        np.testing.assert_allclose(filtered[-1], smoothed[-1], atol=1e-8)
        # Changing the future leaves the past untouched.
        altered = self.observations.copy()
        altered[150:] += 5.0
        refiltered = grid.forward_filter(start, transition, means, variances, altered)
        np.testing.assert_allclose(refiltered[:150], filtered[:150], atol=1e-12)
        self.assertGreater(np.max(np.abs(refiltered[150:] - filtered[150:])), 1e-3)

    def test_state_matching_maps_each_label_to_its_dominant_hidden_state(self):
        labels = np.repeat([0, 1, 2], 10)
        hidden = np.repeat([2, 0, 1], 10)
        hidden[0] = 1  # one disagreement must not change the assignment

        order = grid.match_states(hidden, labels)

        np.testing.assert_array_equal(order, [2, 0, 1])
        self.assertEqual(set(order.tolist()), {0, 1, 2})


class ReplicationTests(unittest.TestCase):
    def test_replication_scores_every_cell_and_archives_probabilities(self):
        cells = ("hmm", "statistics", "statistics_rawsig", "b3", "b3_logsig_shuffled")
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            row = grid.evaluate_replication(
                design="matched_marginal",
                replication=3,
                seed=2024,
                n_paths=(2, 1, 1),
                n_steps=160,
                window_size=20,
                p_switch=0.05,
                n_estimators=5,
                model_jobs=1,
                learners=grid.LEARNERS,
                cells=cells,
                output_dir=output_dir,
            )
            archive = np.load(output_dir / "probabilities" / "replication_03.npz")
            paths = np.load(output_dir / "paths" / "replication_03.npz")

        expected_sizes = {
            "hmm": 3,
            "statistics": 11,
            "statistics_rawsig": 50,
            "b3": 40,
            "b3_logsig_shuffled": 54,
        }
        for learner in grid.LEARNERS:
            for cell in cells:
                prefix = "{}_{}".format(learner, cell)
                self.assertIn(prefix + "_test_ba", row)
                self.assertIn(prefix + "_selected_setting", row)
                self.assertEqual(row[prefix + "_n_features"], expected_sizes[cell])
            self.assertAlmostEqual(
                row["{}_statistics_rawsig_minus_statistics_ba".format(learner)],
                row["{}_statistics_rawsig_test_ba".format(learner)]
                - row["{}_statistics_test_ba".format(learner)],
            )
            self.assertNotIn("{}_b3_logsig_minus_b3_ba".format(learner), row)
        self.assertAlmostEqual(
            row["b3_gbm_minus_random_forest_ba"],
            row["gbm_b3_test_ba"] - row["random_forest_b3_test_ba"],
        )
        self.assertIn("hmm_direct_test_ba", row)
        self.assertEqual(row["design"], "matched_marginal")

        # Only Table-1 cells are archived; the placebo cell is not.
        stored = {key for key in archive.files if key.startswith("test__")}
        self.assertEqual(
            stored,
            {"test__{}__{}".format(learner, cell) for learner in grid.LEARNERS for cell in cells[:-1]},
        )
        probabilities = grid.dequantize(archive["test__logistic__statistics"])
        self.assertEqual(probabilities.shape, (row["n_test_windows"], 3))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-4)
        self.assertEqual(len(archive["test_y"]), row["n_test_windows"])
        self.assertEqual(len(paths["test__test_path_000__log_return"]), 160)
        self.assertEqual(paths["test__test_path_000__regime"].dtype, np.int8)

    def test_summary_reports_positive_counts_for_contrasts_only(self):
        results = pd.DataFrame(
            {
                "design": ["scale_only"] * 4,
                "replication": [1, 2, 3, 4],
                "seed": [1, 2, 3, 4],
                "hmm_direct_test_ba": [0.6, 0.62, 0.61, 0.63],
                "random_forest_statistics_test_ba": [0.70, 0.71, 0.72, 0.73],
                "random_forest_b3_logsig_minus_b3_ba": [0.01, -0.02, 0.03, 0.04],
                "b3_logistic_minus_random_forest_ba": [0.02, 0.01, 0.00, -0.01],
            }
        )

        summary = grid.summarize_results(results).set_index("metric")

        self.assertTrue(np.isnan(summary.loc["BA_random_forest_statistics", "positive_count"]))
        self.assertEqual(summary.loc["Delta_random_forest_b3_logsig_minus_b3", "positive_count"], 3)
        self.assertAlmostEqual(
            summary.loc["Delta_b3_logistic_minus_random_forest", "positive_share"], 0.5
        )
        self.assertAlmostEqual(summary.loc["BA_hmm_direct", "mean"], 0.615)


if __name__ == "__main__":
    unittest.main()


class HmmRefitTests(unittest.TestCase):
    def test_refit_from_archived_paths_reproduces_the_grid_hmm_cell(self):
        import refit_unified_grid_hmm as refit

        cells = ("hmm", "statistics")
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            row = grid.evaluate_replication(
                design="scale_only",
                replication=2,
                seed=77,
                n_paths=(2, 1, 1),
                n_steps=140,
                window_size=20,
                p_switch=0.05,
                n_estimators=5,
                model_jobs=1,
                learners=("logistic", "random_forest"),
                cells=cells,
                output_dir=output_dir,
            )
            splits = refit.load_archived_splits(output_dir / "paths" / "replication_02.npz", 20)
            self.assertEqual(len(splits.train), row["n_train_windows"])
            self.assertEqual(splits.train.X.shape, (row["n_train_windows"], 0))

            updates = refit.refit_replication(
                {
                    "output_dir": str(output_dir),
                    "replication": 2,
                    "seed": 77,
                    "window_size": 20,
                    "n_estimators": 5,
                    "learners": ("logistic", "random_forest"),
                }
            )
            archive = np.load(output_dir / "probabilities" / "replication_02.npz")

        self.assertEqual(updates["hmm_restarts"], grid.HMM_RESTARTS)
        for key in ("hmm_direct_test_ba", "hmm_log_likelihood", "logistic_hmm_test_ba", "random_forest_hmm_test_ba"):
            self.assertAlmostEqual(updates[key], row[key], places=10)
        self.assertEqual(updates["logistic_hmm_selected_setting"], row["logistic_hmm_selected_setting"])
        self.assertIn("test__logistic__hmm", archive.files)
        self.assertIn("test__logistic__statistics", archive.files)

        results = pd.DataFrame([row])
        results["logistic_hmm_test_ba"] = 0.0
        refit.apply_updates(results, updates)
        self.assertAlmostEqual(results.loc[0, "logistic_hmm_test_ba"], row["logistic_hmm_test_ba"])
        self.assertAlmostEqual(
            results.loc[0, "logistic_hmm_minus_statistics_ba"],
            row["logistic_hmm_test_ba"] - row["logistic_statistics_test_ba"],
        )

    def test_hmm_restarts_keep_the_highest_likelihood_fit(self):
        path = _small_path(seed=5, n_steps=400)
        single, single_diagnostics = grid.fit_hmm([path], seed=5, restarts=1)
        best, diagnostics = grid.fit_hmm([path], seed=5, restarts=3)
        self.assertGreaterEqual(diagnostics["hmm_log_likelihood"], single_diagnostics["hmm_log_likelihood"] - 1e-6)
        self.assertEqual(diagnostics["hmm_restarts"], 3.0)
        self.assertIn(diagnostics["hmm_selected_restart"], (0.0, 1.0, 2.0))
