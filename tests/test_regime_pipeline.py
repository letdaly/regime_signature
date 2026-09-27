import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import sys
from unittest import mock

import numpy as np
import pandas as pd

# When this file is launched directly, Python places ``tests/`` rather than the
# project root on sys.path.  Add the root so the sibling module remains
# importable.  Test discovery from the project root already has this path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regime_pipeline import (
    CausalFeatureExtractor,
    DEFAULT_LEVEL_SHIFT_RANGES,
    DEFAULT_N_PATHS,
    FBMStochasticVolatilitySimulator,
    HestonPathSimulator,
    SimulatedPath,
    build_path_level_splits,
    fgn_autocovariance,
    fractional_gaussian_noise,
    parameter_audit_table,
    sample_matched_marginal_parameters,
    sample_regime_parameters,
    sample_scale_only_parameters,
)
import experiment_a_monte_carlo
import experiment_b_monte_carlo
import experiment_b_window_stability
import experiment_fbm_monte_carlo
from experiment_a_monte_carlo import (
    classification_diagnostics,
    diagnostic_tables,
    evaluate_paired_replication as evaluate_experiment_a_replication,
    feature_indices,
    guard_output_directory,
    mean_t_interval,
    replication_seeds,
    summarize_results as summarize_experiment_a,
)
from experiment_b_monte_carlo import classwise_probability_diagnostics
from regime_detection import (
    DETECTOR_CONTRAST_METRICS,
    TIME_SINCE_CHANGE_BUCKETS,
    bucket_of_time_since_change,
    causal_detector,
    detector_metrics,
    stratified_classification,
    time_since_last_change,
)
from regime_pipeline import WindowDataset
from beyond_marginal_benchmark import feature_selections
from experiment_fbm_monte_carlo import (
    build_fbm_splits,
    default_output_dir,
    evaluate_paired_replication as evaluate_fbm_replication,
    scaled_sampler,
    stationary_variance_audit,
)


class LeakageFreePipelineTests(unittest.TestCase):
    def test_window_label_is_endpoint_state(self):
        frame = pd.DataFrame(
            {
                "LogReturn": [0.00, 0.01, -0.01, 0.02, -0.02],
                "Regime": [0, 0, 1, 1, 2],
            }
        )
        path = SimulatedPath("endpoint_test", frame, {}, 1)
        extractor = CausalFeatureExtractor(
            window_size=3, short_window=2, representation="statistics"
        )

        dataset = extractor.extract_path(path)

        np.testing.assert_array_equal(dataset.end_times, [2, 3, 4])
        np.testing.assert_array_equal(dataset.y, [1, 1, 2])
        np.testing.assert_array_equal(dataset.y, frame.loc[dataset.end_times, "Regime"])

    def test_partitions_share_no_simulated_paths(self):
        extractor = CausalFeatureExtractor(
            window_size=20, short_window=5, representation="statistics"
        )
        splits = build_path_level_splits(
            extractor,
            n_paths=(2, 1, 1),
            n_steps=100,
            master_seed=123,
            p_switch=0.05,
        )

        splits.assert_disjoint()
        self.assertTrue(set(splits.train.path_ids).isdisjoint(splits.validation.path_ids))
        self.assertTrue(set(splits.train.path_ids).isdisjoint(splits.test.path_ids))
        self.assertTrue(set(splits.validation.path_ids).isdisjoint(splits.test.path_ids))

    def test_disjointness_audit_rejects_seed_reuse_under_a_new_path_id(self):
        extractor = CausalFeatureExtractor(
            window_size=20, short_window=5, representation="statistics"
        )
        splits = build_path_level_splits(
            extractor, n_paths=(1, 1, 1), n_steps=80, master_seed=124
        )
        validation_path = splits.paths["validation"][0]
        spoofed = replace(validation_path, seed=splits.paths["train"][0].seed)
        tampered = replace(
            splits,
            paths={**splits.paths, "validation": (spoofed,)},
        )

        with self.assertRaisesRegex(AssertionError, "seed reused"):
            tampered.assert_disjoint()

    def test_overlapping_windows_never_cross_path_boundaries(self):
        extractor = CausalFeatureExtractor(
            window_size=20, short_window=5, representation="statistics", stride=1
        )
        splits = build_path_level_splits(
            extractor,
            n_paths=(2, 1, 1),
            n_steps=100,
            master_seed=321,
        )

        expected_windows_per_path = 100 - 20 + 1
        unique_ids, counts = np.unique(splits.train.path_ids, return_counts=True)
        self.assertEqual(len(unique_ids), 2)
        np.testing.assert_array_equal(counts, [expected_windows_per_path] * 2)
        for path_id in unique_ids:
            endpoints = splits.train.end_times[splits.train.path_ids == path_id]
            np.testing.assert_array_equal(endpoints, np.arange(19, 100))

    def test_parameters_are_randomized_across_paths(self):
        extractor = CausalFeatureExtractor(
            window_size=20, short_window=5, representation="statistics"
        )
        splits = build_path_level_splits(
            extractor,
            parameter_ranges=DEFAULT_LEVEL_SHIFT_RANGES,
            n_paths=(3, 1, 1),
            n_steps=80,
            master_seed=456,
        )
        audit = parameter_audit_table(splits)
        train_regime_zero = audit[(audit["split"] == "train") & (audit["regime"] == 0)]

        self.assertEqual(len(train_regime_zero), 3)
        self.assertGreater(train_regime_zero["theta"].nunique(), 1)
        self.assertGreater(train_regime_zero["kappa"].nunique(), 1)

    def test_default_sized_partitions_start_with_all_regimes(self):
        extractor = CausalFeatureExtractor(
            window_size=20, short_window=5, representation="statistics"
        )
        splits = build_path_level_splits(
            extractor,
            n_paths=(3, 3, 3),
            n_steps=40,
            master_seed=654,
            p_switch=0.0,
        )

        for split_name in ("train", "validation", "test"):
            paths = splits.paths[split_name]
            starting_regimes = {int(path.frame["Regime"].iloc[0]) for path in paths}
            self.assertEqual(starting_regimes, {0, 1, 2})

    def test_master_seed_reproduces_data_and_parameter_draws(self):
        extractor = CausalFeatureExtractor(
            window_size=20, short_window=5, representation="statistics"
        )
        first = build_path_level_splits(
            extractor, n_paths=(1, 1, 1), n_steps=80, master_seed=789
        )
        second = build_path_level_splits(
            extractor, n_paths=(1, 1, 1), n_steps=80, master_seed=789
        )

        np.testing.assert_allclose(first.train.X, second.train.X)
        np.testing.assert_array_equal(first.train.y, second.train.y)
        pd.testing.assert_frame_equal(parameter_audit_table(first), parameter_audit_table(second))

    def test_combined_columns_partition_into_paired_representations(self):
        extractor = CausalFeatureExtractor(
            window_size=20,
            short_window=5,
            representation="combined",
            signature_kind="raw",
        )
        splits = build_path_level_splits(
            extractor, n_paths=(1, 1, 1), n_steps=80, master_seed=1357
        )
        selections = feature_indices(splits.train.feature_names)

        self.assertEqual(len(selections["statistics"]), 11)
        self.assertEqual(len(selections["signature"]), 39)
        self.assertEqual(len(selections["combined"]), 50)
        self.assertTrue(
            set(selections["statistics"]).isdisjoint(set(selections["signature"]))
        )
        self.assertEqual(
            set(np.concatenate((selections["statistics"], selections["signature"]))),
            set(selections["combined"]),
        )

    def test_logsignature_representation_has_fourteen_level_three_coordinates(self):
        # Level-3 log-signature of a 3-channel path: 3 + 3 + 8 = 14 coordinates,
        # versus 3 + 9 + 27 = 39 raw signature coordinates.
        extractor = CausalFeatureExtractor(
            window_size=20,
            short_window=5,
            representation="combined",
            signature_kind="log",
        )
        splits = build_path_level_splits(
            extractor, n_paths=(1, 1, 1), n_steps=80, master_seed=1357
        )
        selections = feature_indices(splits.train.feature_names)

        self.assertEqual(len(selections["statistics"]), 11)
        self.assertEqual(len(selections["signature"]), 14)
        self.assertEqual(len(selections["combined"]), 25)
        self.assertTrue(
            all(
                splits.train.feature_names[index].startswith("logsignature_")
                for index in selections["signature"]
            )
        )
        self.assertTrue(np.isfinite(splits.train.X).all())
        # The first three log-signature coordinates are the channel increments
        # of the window path, i.e. level one of the raw signature.
        raw = CausalFeatureExtractor(
            window_size=20, short_window=5, representation="signature", signature_kind="raw"
        ).extract_path(splits.paths["train"][0])
        log = CausalFeatureExtractor(
            window_size=20, short_window=5, representation="signature", signature_kind="log"
        ).extract_path(splits.paths["train"][0])
        np.testing.assert_allclose(log.X[:, :3], raw.X[:, :3], rtol=1e-10, atol=1e-12)

    def test_rich_baseline_hierarchy_is_nested_and_complete(self):
        extractor = CausalFeatureExtractor(
            window_size=20,
            short_window=5,
            representation="combined",
            signature_kind="raw",
            statistics_level="rich",
        )
        splits = build_path_level_splits(
            extractor, n_paths=(1, 1, 1), n_steps=50, master_seed=246
        )
        selections = feature_selections(splits.train.feature_names)

        self.assertEqual(len(selections["b1"]), 14)
        self.assertEqual(len(selections["b2"]), 29)
        self.assertEqual(len(selections["b3"]), 40)
        self.assertEqual(len(selections["signature"]), 39)
        self.assertEqual(len(selections["b3_signature"]), 79)
        self.assertTrue(set(selections["b1"]).issubset(selections["b2"]))
        self.assertTrue(set(selections["b2"]).issubset(selections["b3"]))

    def test_signature_path_is_origin_anchored_with_one_increment_per_return(self):
        # x_j = (j/W, sum_{a<=j} r_a, sum_{a<=j} |r_a|) for j = 0..W: W + 1 points,
        # the first at the origin, so every return of the window is an increment.
        returns = np.array([0.01, -0.02, 0.005, 0.03])
        path = CausalFeatureExtractor(window_size=4, short_window=2)._path(returns)
        self.assertEqual(path.shape, (5, 3))
        np.testing.assert_allclose(path[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(path[:, 0], np.arange(5) / 4.0)
        np.testing.assert_allclose(path[:, 1], np.concatenate(([0.0], np.cumsum(returns))))
        np.testing.assert_allclose(
            path[:, 2], np.concatenate(([0.0], np.cumsum(np.abs(returns))))
        )
        self.assertEqual(CausalFeatureExtractor.PATH_CONVENTION, "origin_anchored_w_increments")

    def test_first_return_of_the_window_reaches_the_signature(self):
        # Regression test for the pre-2026-09-19 path, which started at
        # (0, r_1, |r_1|) and therefore never exposed r_1 to the signature.
        rng = np.random.default_rng(5)
        returns = rng.normal(scale=0.01, size=50)
        altered = returns.copy()
        altered[0] += 0.05
        for signature_kind in ("raw", "log"):
            extractor = CausalFeatureExtractor(
                window_size=50, representation="signature", signature_kind=signature_kind
            )
            original = extractor._features(returns)
            changed = extractor._features(altered)
            self.assertGreater(np.max(np.abs(original - changed)), 1e-6, signature_kind)
            # Level one of either signature is the total increment of each channel.
            np.testing.assert_allclose(
                original[:3], [1.0, returns.sum(), np.abs(returns).sum()], rtol=1e-10
            )

    def test_shuffle_placebo_preserves_labels_but_changes_path_order_features(self):
        frame = pd.DataFrame(
            {
                "LogReturn": np.linspace(-0.03, 0.04, 40) ** 3,
                "Regime": np.repeat([0, 1], 20),
            }
        )
        path = SimulatedPath("shuffle_test", frame, {}, 77)
        original = CausalFeatureExtractor(
            window_size=20,
            short_window=5,
            representation="signature",
            increment_order="original",
        ).extract_path(path)
        shuffled_extractor = CausalFeatureExtractor(
            window_size=20,
            short_window=5,
            representation="signature",
            increment_order="shuffle",
            order_seed=12,
        )
        shuffled_first = shuffled_extractor.extract_path(path)
        shuffled_second = CausalFeatureExtractor(
            window_size=20,
            short_window=5,
            representation="signature",
            increment_order="shuffle",
            order_seed=12,
        ).extract_path(path)

        np.testing.assert_array_equal(original.y, shuffled_first.y)
        np.testing.assert_array_equal(original.end_times, shuffled_first.end_times)
        np.testing.assert_allclose(shuffled_first.X, shuffled_second.X)
        self.assertFalse(np.allclose(original.X, shuffled_first.X))
        np.testing.assert_allclose(
            np.sort(frame["LogReturn"].to_numpy()[:20]),
            np.sort(shuffled_extractor._ordered_returns(frame["LogReturn"].to_numpy()[:20], 9)),
        )

    def test_future_returns_cannot_change_earlier_causal_features(self):
        base_returns = np.linspace(-0.02, 0.03, 60)
        changed_returns = base_returns.copy()
        changed_returns[40:] = 100.0
        regimes = np.repeat([0, 1, 2], 20)
        extractor = CausalFeatureExtractor(
            window_size=20,
            short_window=5,
            representation="combined",
            statistics_level="rich",
        )
        first = extractor.extract_path(
            SimulatedPath(
                "causal_future_test",
                pd.DataFrame({"LogReturn": base_returns, "Regime": regimes}),
                {},
                88,
            )
        )
        second = extractor.extract_path(
            SimulatedPath(
                "causal_future_test",
                pd.DataFrame({"LogReturn": changed_returns, "Regime": regimes}),
                {},
                88,
            )
        )
        unaffected = first.end_times < 40
        np.testing.assert_allclose(first.X[unaffected], second.X[unaffected])

    def test_paired_mean_confidence_interval_uses_standard_error(self):
        summary = mean_t_interval([0.01, 0.02, 0.03, 0.04])

        self.assertAlmostEqual(summary["mean"], 0.025)
        self.assertAlmostEqual(
            summary["standard_error"], np.std([0.01, 0.02, 0.03, 0.04], ddof=1) / 2
        )
        self.assertLess(summary["ci_low"], summary["mean"])
        self.assertGreater(summary["ci_high"], summary["mean"])

    def test_replication_seeds_are_unique_and_reproducible(self):
        first = replication_seeds(2468, 50)
        second = replication_seeds(2468, 50)

        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 50)

    def test_classification_diagnostics_use_fixed_regime_order(self):
        truth = np.array([0, 0, 1, 1, 2, 2])
        prediction = np.array([0, 1, 1, 1, 0, 2])

        result = classification_diagnostics(truth, prediction, "combined_test")

        self.assertAlmostEqual(result["combined_test_recall_0"], 0.5)
        self.assertAlmostEqual(result["combined_test_recall_1"], 1.0)
        self.assertAlmostEqual(result["combined_test_recall_2"], 0.5)
        self.assertEqual(result["combined_test_cm_0_0"], 1)
        self.assertEqual(result["combined_test_cm_0_1"], 1)
        self.assertEqual(result["combined_test_cm_2_0"], 1)
        self.assertEqual(result["combined_test_cm_2_2"], 1)

    def test_matched_marginal_sampler_preserves_cir_stationary_law(self):
        parameters = sample_matched_marginal_parameters(np.random.default_rng(2026))

        theta = [parameters[regime]["theta"] for regime in range(3)]
        variance_ratio = [
            parameters[regime]["sigma"] ** 2 / parameters[regime]["kappa"]
            for regime in range(3)
        ]
        mu = [parameters[regime]["mu"] for regime in range(3)]
        rho = [parameters[regime]["rho"] for regime in range(3)]

        np.testing.assert_allclose(theta, np.repeat(theta[0], 3), rtol=0, atol=1e-14)
        np.testing.assert_allclose(
            variance_ratio, np.repeat(variance_ratio[0], 3), rtol=0, atol=1e-14
        )
        np.testing.assert_allclose(mu, np.repeat(mu[0], 3), rtol=0, atol=1e-14)
        np.testing.assert_allclose(rho, np.repeat(rho[0], 3), rtol=0, atol=1e-14)
        self.assertLess(parameters[0]["kappa"], parameters[1]["kappa"])
        self.assertLess(parameters[1]["kappa"], parameters[2]["kappa"])

    def test_scale_only_sampler_changes_only_variance_scale(self):
        parameters = sample_scale_only_parameters(np.random.default_rng(2026))

        for name in ("mu", "kappa", "rho"):
            values = [parameters[regime][name] for regime in range(3)]
            np.testing.assert_allclose(values, np.repeat(values[0], 3), rtol=0, atol=0)

        theta = np.array([parameters[regime]["theta"] for regime in range(3)])
        self.assertTrue(np.all(np.diff(theta) > 0))
        scale_coefficients = np.array(
            [
                parameters[regime]["sigma"]
                / np.sqrt(parameters[regime]["kappa"] * parameters[regime]["theta"])
                for regime in range(3)
            ]
        )
        np.testing.assert_allclose(
            scale_coefficients,
            np.repeat(scale_coefficients[0], 3),
            rtol=0,
            atol=1e-14,
        )

        # V/theta has a regime-invariant stationary Gamma law.
        normalized_shape = np.array(
            [
                2 * parameters[regime]["kappa"] * parameters[regime]["theta"]
                / parameters[regime]["sigma"] ** 2
                for regime in range(3)
            ]
        )
        normalized_scale = np.array(
            [
                parameters[regime]["sigma"] ** 2
                / (2 * parameters[regime]["kappa"] * parameters[regime]["theta"])
                for regime in range(3)
            ]
        )
        np.testing.assert_allclose(normalized_shape, normalized_shape[0])
        np.testing.assert_allclose(normalized_scale, normalized_scale[0])

    def test_matched_marginal_paths_use_stationary_initial_variance(self):
        extractor = CausalFeatureExtractor(
            window_size=20, short_window=5, representation="statistics"
        )
        splits = build_path_level_splits(
            extractor,
            parameter_sampler=sample_matched_marginal_parameters,
            stationary_initial_variance=True,
            simulation_substeps=4,
            n_paths=(3, 3, 3),
            n_steps=40,
            master_seed=987,
            p_switch=0.0,
        )
        audit = parameter_audit_table(splits)
        for path_id, group in audit.groupby("path_id"):
            np.testing.assert_allclose(group["theta"], group["theta"].iloc[0])
            ratios = group["sigma"] ** 2 / group["kappa"]
            np.testing.assert_allclose(ratios, ratios.iloc[0], rtol=0, atol=1e-14)

    def test_causal_detector_requires_confirmed_predictions(self):
        probabilities = np.array(
            [
                [0.9, 0.1, 0.0],
                [0.1, 0.9, 0.0],
                [0.1, 0.9, 0.0],
                [0.1, 0.9, 0.0],
            ]
        )
        decisions = causal_detector(
            probabilities, alpha=1.0, threshold=0.5, confirmations=3
        )
        np.testing.assert_array_equal(decisions, [0, 0, 0, 1])

    def test_classwise_diagnostics_detect_fast_overprediction(self):
        truth = np.array([0, 0, 1, 1, 2, 2])
        probabilities = np.array(
            [
                [0.80, 0.10, 0.10],
                [0.30, 0.20, 0.50],
                [0.10, 0.80, 0.10],
                [0.10, 0.30, 0.60],
                [0.10, 0.10, 0.80],
                [0.05, 0.15, 0.80],
            ]
        )
        prediction = np.argmax(probabilities, axis=1)

        diagnostics = classwise_probability_diagnostics(
            truth, prediction, probabilities, "candidate_test", n_bins=5
        )

        self.assertAlmostEqual(diagnostics["candidate_test_prevalence_2"], 1 / 3)
        self.assertAlmostEqual(
            diagnostics["candidate_test_predicted_proportion_2"], 4 / 6
        )
        self.assertAlmostEqual(
            diagnostics["candidate_test_prediction_excess_2"], 1 / 3
        )
        self.assertAlmostEqual(diagnostics["candidate_test_precision_2"], 0.5)
        self.assertAlmostEqual(diagnostics["candidate_test_recall_2"], 1.0)
        self.assertAlmostEqual(diagnostics["candidate_test_f1_2"], 2 / 3)
        self.assertGreater(diagnostics["candidate_test_roc_auc_2"], 0.5)
        self.assertGreater(diagnostics["candidate_test_pr_auc_2"], 1 / 3)
        self.assertEqual(
            sum(
                diagnostics[f"candidate_test_calibration_2_{index}_count"]
                for index in range(5)
            ),
            len(truth),
        )

    def test_detector_metrics_match_only_causal_same_state_events(self):
        dataset = type("Dataset", (), {})()
        dataset.y = np.array([0, 0, 1, 1, 2, 2])
        dataset.path_ids = np.repeat("test_path", 6)
        dataset.end_times = np.arange(6)
        probabilities = np.eye(3)[np.array([0, 0, 0, 1, 1, 2])]

        metrics = detector_metrics(
            dataset,
            probabilities,
            alpha=1.0,
            threshold=0.5,
            confirmations=1,
            tolerance=2,
        )

        self.assertEqual(metrics["n_true_changes"], 2)
        self.assertEqual(metrics["n_matched_changes"], 2)
        self.assertEqual(metrics["n_false_switches"], 0)
        self.assertEqual(metrics["detection_probability"], 1.0)
        self.assertEqual(metrics["mean_detection_delay"], 1.0)

    def test_time_since_change_is_measured_from_the_first_observation_of_the_new_regime(self):
        regimes = np.array([0] * 60 + [1] * 30 + [2] * 40)  # changes at t = 60 and t = 90
        path = SimulatedPath(
            "p", pd.DataFrame({"LogReturn": np.zeros(130), "Regime": regimes}), {}, 1
        )
        end_times = np.arange(49, 130)
        dataset = WindowDataset(
            X=np.zeros((len(end_times), 1)),
            y=regimes[end_times],
            path_ids=np.repeat("p", len(end_times)),
            end_times=end_times,
            feature_names=("f",),
        )

        distances = time_since_last_change(dataset, [path])

        self.assertTrue(np.isinf(distances[end_times < 60]).all())
        self.assertEqual(distances[end_times == 60][0], 0)
        self.assertEqual(distances[end_times == 89][0], 29)
        self.assertEqual(distances[end_times == 90][0], 0)
        self.assertEqual(distances[end_times == 129][0], 39)
        buckets = bucket_of_time_since_change(distances)
        counts = dict(zip(*np.unique(buckets, return_counts=True)))
        self.assertEqual(counts, {"0_24": 50, "25_49": 20, "100_plus": 11})

        # A predictor that is right only once the window is mature.
        prediction = np.where(distances >= 25, dataset.y, (dataset.y + 1) % 3)
        stratified = stratified_classification(dataset.y, prediction, distances, "x")
        self.assertEqual(stratified["x_ba_since_0_24"], 0.0)
        self.assertEqual(stratified["x_ba_since_25_49"], 1.0)
        self.assertTrue(np.isnan(stratified["x_ba_since_50_99"]))
        self.assertEqual(stratified["x_n_since_50_99"], 0)
        self.assertEqual(stratified["x_ba_since_100_plus"], 1.0)
        self.assertEqual(
            sum(stratified["x_n_since_" + name] for name, _, _ in TIME_SINCE_CHANGE_BUCKETS),
            len(dataset),
        )

    def test_experiment_a_reports_validation_tuned_detector_metrics(self):
        common = dict(
            replication=1,
            seed=2026,
            n_paths=(2, 1, 1),
            n_steps=160,
            window_size=20,
            p_switch=0.05,
            n_estimators=10,
            event_tolerance=30,
        )
        with_detector = evaluate_experiment_a_replication(**common)
        without_detector = evaluate_experiment_a_replication(
            **common, evaluate_detector=False
        )

        # Classification metrics are unaffected by the detector stage.
        for column, value in without_detector.items():
            if isinstance(value, float) and np.isnan(value):
                self.assertTrue(np.isnan(with_detector[column]), column)
            else:
                self.assertAlmostEqual(with_detector[column], value, places=12)
        self.assertNotIn("combined_test_boundary_f1", without_detector)
        # Time-since-change buckets partition the test windows for every representation.
        for representation in ("statistics", "signature", "combined"):
            bucket_total = sum(
                with_detector["{}_test_n_since_{}".format(representation, name)]
                for name, _, _ in TIME_SINCE_CHANGE_BUCKETS
            )
            self.assertEqual(bucket_total, with_detector["n_test_windows"])
        self.assertIn("combined_minus_statistics_ba_since_100_plus", with_detector)
        for representation in ("statistics", "signature", "combined"):
            self.assertIn(with_detector[representation + "_detector_alpha"], (0.2, 0.5, 1.0))
            self.assertGreaterEqual(with_detector[representation + "_test_false_switches_per_1000"], 0.0)
            self.assertGreaterEqual(with_detector[representation + "_test_n_true_changes"], 0)
        for metric in DETECTOR_CONTRAST_METRICS:
            contrast = with_detector["combined_minus_statistics_" + metric]
            if np.isfinite(contrast):
                self.assertAlmostEqual(
                    contrast,
                    with_detector["combined_test_" + metric]
                    - with_detector["statistics_test_" + metric],
                )

        results = pd.DataFrame([with_detector, {**with_detector, "replication": 2}])
        summary = summarize_experiment_a(results).set_index("metric")
        self.assertIn("Boundary_F1_combined", summary.index)
        self.assertIn("Delta_False_switches_per_1000_combined_minus_statistics", summary.index)
        # Delay rows exist only when at least one change was matched; NaN
        # delays are dropped rather than averaged.
        self.assertEqual(
            "Mean_delay_combined" in summary.index,
            np.isfinite(with_detector["combined_test_mean_detection_delay"]),
        )
        _, diagnostics = diagnostic_tables(results)
        self.assertIn("false_switches_per_1000", diagnostics.columns)
        # Classification-only tables (the frozen study) still summarize.
        legacy = summarize_experiment_a(pd.DataFrame([without_detector] * 2))
        self.assertEqual(
            legacy["metric"].tolist()[:3], ["BA_statistics", "BA_signature", "BA_combined"]
        )
        self.assertNotIn("Boundary_F1_combined", legacy["metric"].tolist())

    def test_time_since_last_change_treats_a_path_without_switches_as_mature(self):
        frame = pd.DataFrame({"LogReturn": np.zeros(30), "Regime": np.zeros(30, dtype=int)})
        path = SimulatedPath("no_switch", frame, {}, 3)
        dataset = CausalFeatureExtractor(window_size=10, short_window=3).extract_path(path)
        distances = time_since_last_change(dataset, [path])
        self.assertTrue(np.isinf(distances).all())
        self.assertTrue((bucket_of_time_since_change(distances) == "100_plus").all())

    def test_logistic_learner_is_a_paired_second_column_of_the_main_grid(self):
        common = dict(
            replication=1,
            seed=2027,
            n_paths=(2, 1, 1),
            n_steps=160,
            window_size=20,
            p_switch=0.05,
            n_estimators=10,
            event_tolerance=30,
        )
        both = evaluate_experiment_a_replication(**common)
        forest_only = evaluate_experiment_a_replication(
            **common, learners=("random_forest",)
        )
        # The forest keeps its historical column names and values; the
        # logistic learner adds prefixed columns on the same windows.
        for column, value in forest_only.items():
            if column == "elapsed_seconds":
                continue
            if isinstance(value, float) and np.isnan(value):
                self.assertTrue(np.isnan(both[column]), column)
            else:
                self.assertAlmostEqual(both[column], value, places=12, msg=column)
        self.assertNotIn("logistic_combined_test_ba", forest_only)
        for representation in ("statistics", "signature", "combined"):
            cell = "logistic_" + representation
            self.assertEqual(both[cell + "_n_features"], both[representation + "_n_features"])
            self.assertIn(both[cell + "_selected_c"], experiment_a_monte_carlo.LOGISTIC_C_GRID)
            self.assertIn(both[cell + "_detector_alpha"], (0.2, 0.5, 1.0))
            self.assertAlmostEqual(
                both["logistic_minus_random_forest_{}_ba".format(representation)],
                both[cell + "_test_ba"] - both[representation + "_test_ba"],
            )
        self.assertAlmostEqual(
            both["logistic_combined_minus_statistics_ba"],
            both["logistic_combined_test_ba"] - both["logistic_statistics_test_ba"],
        )
        for metric in DETECTOR_CONTRAST_METRICS:
            contrast = both["logistic_combined_minus_statistics_" + metric]
            if np.isfinite(contrast):
                self.assertAlmostEqual(
                    contrast,
                    both["logistic_combined_test_" + metric]
                    - both["logistic_statistics_test_" + metric],
                )

        results = pd.DataFrame([both, {**both, "replication": 2}])
        summary = summarize_experiment_a(results).set_index("metric")
        for label in (
            "BA_logistic_combined",
            "Delta_logistic_combined_minus_statistics",
            "BA_since_100_plus_logistic_statistics",
            "Boundary_F1_logistic_combined",
            "Delta_BA_logistic_minus_random_forest_combined",
        ):
            self.assertIn(label, summary.index)
        _, diagnostics = diagnostic_tables(results)
        self.assertEqual(
            sorted(diagnostics["learner"].unique()), ["logistic", "random_forest"]
        )
        self.assertEqual(len(diagnostics), 2 * 2 * 3)
        self.assertTrue(
            diagnostics.loc[diagnostics.learner == "random_forest", "selected_c"].isna().all()
        )

    def test_experiment_b_scores_both_learners_with_classwise_diagnostics(self):
        row = experiment_b_monte_carlo.evaluate_paired_replication(
            replication=1,
            seed=2028,
            n_paths=(2, 1, 1),
            n_steps=160,
            window_size=20,
            p_switch=0.05,
            n_estimators=10,
            event_tolerance=30,
        )
        for cell in ("combined", "logistic_combined"):
            self.assertIn(cell + "_test_roc_auc_2", row)
            self.assertIn(cell + "_test_calibration_0_0_count", row)
            self.assertAlmostEqual(
                row[cell + "_minus_statistics_recall_2"],
                row[cell + "_test_recall_2"]
                - row[cell.replace("combined", "statistics") + "_test_recall_2"],
            )
        self.assertIn("logistic_minus_random_forest_signature_ba", row)
        summary = experiment_b_monte_carlo.summarize_results(
            pd.DataFrame([row, {**row, "replication": 2}])
        ).set_index("metric")
        for label in (
            "Delta_BA_combined_minus_statistics",
            "Delta_mean_delay_combined_minus_statistics",
            "Delta_BA_logistic_combined_minus_statistics",
            "Delta_boundary_F1_logistic_combined_minus_statistics",
        ):
            self.assertIn(label, summary.index)
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            experiment_b_monte_carlo.save_outputs(
                pd.DataFrame([row, {**row, "replication": 2}]), output_dir
            )
            classwise = pd.read_csv(output_dir / "classwise_contrasts.csv")
            self.assertEqual(sorted(classwise["learner"].unique()), ["logistic", "random_forest"])
            calibration = pd.read_csv(output_dir / "calibration_bins.csv")
            self.assertIn("learner", calibration.columns)
            diagnostics = pd.read_csv(output_dir / "test_diagnostics.csv")
            self.assertEqual(len(diagnostics), 2 * 2 * 3)

    def test_truncation_ablation_levels_nest_and_level_three_reproduces_the_main_grid(self):
        import experiment_truncation_ablation as ablation

        common = dict(
            seed=2029, n_paths=(2, 1, 1), n_steps=160, window_size=20, p_switch=0.05, n_estimators=10
        )
        row = ablation.evaluate_paired_replication(
            replication=1, design="scale_only", levels=(2, 3, 4), learners=("random_forest",), **common
        )
        # Column counts: Statistics 11, log-signature 6 / 14 / 32, noise control matches each level.
        self.assertEqual(row["statistics_n_features"], 11)
        for level, length in ((2, 6), (3, 14), (4, 32)):
            self.assertEqual(row["signature_{}_n_features".format(level)], length)
            self.assertEqual(row["combined_{}_n_features".format(level)], 11 + length)
            self.assertEqual(row["noise_{}_n_features".format(level)], 11 + length)
        # Level 3 of the ablation is literally the main-grid Combined cell: same
        # paths, same columns (log-signature levels nest as prefixes), same seed.
        main = evaluate_experiment_a_replication(
            replication=1, event_tolerance=30, evaluate_detector=False,
            learners=("random_forest",), **common
        )
        self.assertAlmostEqual(row["combined_3_test_ba"], main["combined_test_ba"], places=12)
        self.assertAlmostEqual(row["signature_3_test_ba"], main["signature_test_ba"], places=12)
        self.assertAlmostEqual(row["statistics_test_ba"], main["statistics_test_ba"], places=12)
        self.assertAlmostEqual(
            row["combined_3_minus_statistics_ba"], main["combined_minus_statistics_ba"], places=12
        )
        self.assertAlmostEqual(
            row["combined_4_minus_combined_3_ba"], row["combined_4_test_ba"] - row["combined_3_test_ba"]
        )
        self.assertIn("combined_2_minus_noise_2_ba_since_100_plus", row)
        self.assertNotIn("logistic_combined_3_test_ba", row)

        # The noise control is reproducible from the seed and independent across partitions.
        first = ablation.build_ablation_splits("scale_only", 2029, (2, 1, 1), 160, 20, 0.05, (2, 3, 4))
        second = ablation.build_ablation_splits("scale_only", 2029, (2, 1, 1), 160, 20, 0.05, (2, 3, 4))
        noise_columns = [i for i, n in enumerate(first.train.feature_names) if n.startswith("noise_")]
        self.assertEqual(len(noise_columns), 32)
        np.testing.assert_array_equal(first.train.X[:, noise_columns], second.train.X[:, noise_columns])
        self.assertFalse(
            np.array_equal(first.train.X[:5, noise_columns], first.test.X[:5, noise_columns])
        )

        results = pd.DataFrame([row, {**row, "replication": 2}])
        summary = ablation.summarize_results(results, (2, 3, 4)).set_index("metric")
        for label in ("BA_combined_4", "Delta_combined_3_minus_statistics", "Delta_combined_3_minus_noise_3",
                      "Delta_combined_4_minus_combined_3", "Delta_BA_since_100_plus_combined_2_minus_noise_2"):
            self.assertIn(label, summary.index)
        self.assertIn("positive_share", summary.columns)
        diagnostics = ablation.diagnostic_table(results, (2, 3, 4))
        self.assertEqual(len(diagnostics), 2 * 10)


SINGLE_REGIME = {0: {"mu": 0.02, "kappa": 4.0, "theta": 0.08, "sigma": 0.30, "rho": -0.70}}


def _lag_one_autocorrelation(values: np.ndarray) -> float:
    centered = values - values.mean()
    return float(np.dot(centered[:-1], centered[1:]) / np.dot(centered, centered))


class FractionalGaussianNoiseTests(unittest.TestCase):
    def test_autocovariance_matches_closed_form_and_is_white_at_half(self):
        rough = fgn_autocovariance(3, 0.2)
        self.assertAlmostEqual(rough[0], 1.0)
        self.assertAlmostEqual(rough[1], 0.5 * (2 ** 0.4 - 2.0))
        self.assertAlmostEqual(rough[2], 0.5 * (3 ** 0.4 - 2 * 2 ** 0.4 + 1.0))
        np.testing.assert_allclose(fgn_autocovariance(5, 0.5), [1, 0, 0, 0, 0, 0], atol=0)
        for bad in (0.0, 1.0, -0.2, np.nan):
            with self.assertRaises(ValueError):
                fgn_autocovariance(3, bad)

    def test_hosking_map_is_identity_at_half_and_causal_otherwise(self):
        rng = np.random.default_rng(11)
        white = rng.standard_normal(64)

        np.testing.assert_array_equal(fractional_gaussian_noise(white, 0.5), white)

        rough = fractional_gaussian_noise(white, 0.2)
        perturbed = white.copy()
        perturbed[40:] += 5.0
        rough_perturbed = fractional_gaussian_noise(perturbed, 0.2)
        np.testing.assert_allclose(rough[:40], rough_perturbed[:40])
        self.assertFalse(np.allclose(rough[40:], rough_perturbed[40:]))
        np.testing.assert_allclose(rough, fractional_gaussian_noise(white, 0.2))

    def test_hosking_map_has_exactly_the_fgn_covariance(self):
        # The map is linear and lower triangular: noise = L z.  Recover L from
        # unit vectors and check L L^T against the Toeplitz fGn covariance.
        n = 48
        for hurst in (0.2, 0.35, 0.8):
            columns = [fractional_gaussian_noise(np.eye(n)[:, j], hurst) for j in range(n)]
            lower = np.column_stack(columns)
            self.assertTrue(np.allclose(np.triu(lower, k=1), 0.0))
            gamma = fgn_autocovariance(n - 1, hurst)
            toeplitz = gamma[np.abs(np.subtract.outer(np.arange(n), np.arange(n)))]
            # einsum rather than "@": Apple Accelerate under NumPy 2.0 raises a
            # spurious divide-by-zero RuntimeWarning inside matmul.
            covariance = np.einsum("ik,jk->ij", lower, lower)
            np.testing.assert_allclose(covariance, toeplitz, atol=1e-10)


class FBMStochasticVolatilitySimulatorTests(unittest.TestCase):
    def test_seed_reproduces_path_and_frame_matches_heston_schema(self):
        parameters = sample_matched_marginal_parameters(np.random.default_rng(3))
        simulator = FBMStochasticVolatilitySimulator(hurst=0.35, substeps=2)
        first = simulator.simulate("f", parameters, 120, seed=7, p_switch=0.05, min_dwell=10)
        second = simulator.simulate("f", parameters, 120, seed=7, p_switch=0.05, min_dwell=10)
        heston = HestonPathSimulator(substeps=2).simulate(
            "h", parameters, 120, seed=7, p_switch=0.05, min_dwell=10
        )

        pd.testing.assert_frame_equal(first.frame, second.frame)
        self.assertEqual(list(first.frame.columns), list(heston.frame.columns))
        self.assertEqual(first.frame["LogReturn"].iloc[0], 0.0)
        self.assertTrue((first.frame["Variance"] >= 0.0).all())
        self.assertFalse(np.allclose(first.frame["Price"], heston.frame["Price"]))
        for regime in range(3):
            self.assertEqual(first.regime_parameters[regime]["hurst"], 0.35)
        self.assertNotIn("hurst", heston.regime_parameters[0])

    def test_rough_variance_increments_are_antipersistent_but_returns_are_not(self):
        acf_variance = {}
        for hurst in (0.2, 0.5):
            frame = FBMStochasticVolatilitySimulator(hurst=hurst).simulate(
                "d", SINGLE_REGIME, 6000, seed=5, p_switch=0.0, v0=None
            ).frame
            acf_variance[hurst] = _lag_one_autocorrelation(np.diff(frame["Variance"].to_numpy()))
            returns = frame["LogReturn"].to_numpy()[1:]
            self.assertLess(abs(_lag_one_autocorrelation(returns)), 0.05)

        # Near the theoretical fGn value 0.5 * (2^0.4 - 2) = -0.34 for H = 0.2,
        # versus roughly -kappa * dt / 2 = -0.008 for the Brownian case.
        self.assertLess(acf_variance[0.2], -0.25)
        self.assertGreater(acf_variance[0.5], -0.05)

    def test_half_hurst_recovers_cir_stationary_moments(self):
        frame = FBMStochasticVolatilitySimulator(hurst=0.5).simulate(
            "cir", SINGLE_REGIME, 20000, seed=13, p_switch=0.0, v0=None
        ).frame
        parameters = SINGLE_REGIME[0]
        stationary_mean = parameters["theta"]
        stationary_variance = (
            parameters["sigma"] ** 2 * parameters["theta"] / (2.0 * parameters["kappa"])
        )

        self.assertAlmostEqual(frame["Variance"].mean(), stationary_mean, delta=0.1 * stationary_mean)
        self.assertAlmostEqual(
            frame["Variance"].var(), stationary_variance, delta=0.3 * stationary_variance
        )

    def test_per_regime_hurst_overrides_simulator_default(self):
        parameters = {
            0: {**SINGLE_REGIME[0], "hurst": 0.2},
            1: {**SINGLE_REGIME[0]},
        }
        simulator = FBMStochasticVolatilitySimulator(hurst=0.45)
        path = simulator.simulate("mix", parameters, 60, seed=21, p_switch=0.0)

        self.assertEqual(path.regime_parameters[0]["hurst"], 0.2)
        self.assertEqual(path.regime_parameters[1]["hurst"], 0.45)
        with self.assertRaisesRegex(ValueError, "hurst"):
            HestonPathSimulator().simulate("h", parameters, 60, seed=21)
        with self.assertRaises(ValueError):
            FBMStochasticVolatilitySimulator(hurst=1.0)

    def test_build_path_level_splits_switches_simulator_on_hurst(self):
        extractor = CausalFeatureExtractor(
            window_size=20, short_window=5, representation="statistics"
        )
        heston = build_path_level_splits(
            extractor, n_paths=(1, 1, 1), n_steps=60, master_seed=31
        )
        rough = build_path_level_splits(
            extractor, n_paths=(1, 1, 1), n_steps=60, master_seed=31, hurst=0.2
        )
        heston_audit = parameter_audit_table(heston)
        rough_audit = parameter_audit_table(rough)

        rough.assert_disjoint()
        self.assertNotIn("hurst", heston_audit.columns)
        self.assertTrue((rough_audit["hurst"] == 0.2).all())
        # Same master seed gives the same parameter draws; only dynamics differ.
        shared = ["split", "path_id", "path_seed", "regime", "mu", "kappa", "theta", "sigma", "rho"]
        pd.testing.assert_frame_equal(heston_audit[shared], rough_audit[shared])
        self.assertFalse(np.allclose(heston.train.X, rough.train.X))

    def test_roughness_shift_ranges_are_sampled_per_regime(self):
        ranges = {
            regime: {**DEFAULT_LEVEL_SHIFT_RANGES[regime], "hurst": bounds}
            for regime, bounds in {0: (0.15, 0.25), 1: (0.45, 0.5), 2: (0.7, 0.8)}.items()
        }
        parameters = sample_regime_parameters(ranges, np.random.default_rng(41))
        self.assertLess(parameters[0]["hurst"], parameters[1]["hurst"])
        self.assertLess(parameters[1]["hurst"], parameters[2]["hurst"])

        invalid = {**ranges, 0: {**ranges[0], "hurst": (0.0, 0.2)}}
        with self.assertRaisesRegex(ValueError, "hurst"):
            sample_regime_parameters(invalid, np.random.default_rng(41))


class FBMRowRunnerTests(unittest.TestCase):
    def test_sigma_scale_preserves_both_scenario_invariants(self):
        scale_only = scaled_sampler(sample_scale_only_parameters, 2.0)(np.random.default_rng(9))
        base = sample_scale_only_parameters(np.random.default_rng(9))
        for regime in range(3):
            self.assertAlmostEqual(scale_only[regime]["sigma"], 2.0 * base[regime]["sigma"])
        coefficients = [
            scale_only[r]["sigma"] / np.sqrt(scale_only[r]["kappa"] * scale_only[r]["theta"])
            for r in range(3)
        ]
        np.testing.assert_allclose(coefficients, coefficients[0])

        matched = scaled_sampler(sample_matched_marginal_parameters, 1.5)(np.random.default_rng(9))
        ratios = [matched[r]["sigma"] ** 2 / matched[r]["kappa"] for r in range(3)]
        np.testing.assert_allclose(ratios, ratios[0])
        self.assertIs(scaled_sampler(sample_scale_only_parameters, 1.0), sample_scale_only_parameters)
        with self.assertRaises(ValueError):
            scaled_sampler(sample_scale_only_parameters, 0.0)

    def test_fbm_row_pairs_parameter_families_with_the_heston_row(self):
        extractor = CausalFeatureExtractor(
            window_size=20, short_window=5, representation="combined", signature_kind="raw"
        )
        heston = build_path_level_splits(
            extractor,
            parameter_sampler=sample_scale_only_parameters,
            n_paths=(1, 1, 1),
            n_steps=60,
            master_seed=20260827,
            min_dwell=20,
            stationary_initial_variance=True,
        )
        rough = build_fbm_splits(
            seed=20260827,
            scenario="scale_only",
            hurst=0.2,
            sigma_scale=1.0,
            n_paths=(1, 1, 1),
            n_steps=60,
            window_size=20,
            p_switch=0.002,
            substeps=1,
        )
        shared = ["split", "path_id", "path_seed", "regime", "mu", "kappa", "theta", "sigma", "rho"]
        pd.testing.assert_frame_equal(
            parameter_audit_table(heston)[shared], parameter_audit_table(rough)[shared]
        )
        self.assertTrue((parameter_audit_table(rough)["hurst"] == 0.2).all())

    def test_fbm_replication_row_records_design_and_detector_columns(self):
        row = evaluate_fbm_replication(
            replication=1,
            seed=77,
            scenario="scale_only",
            hurst=0.35,
            sigma_scale=1.0,
            n_paths=(2, 1, 1),
            n_steps=160,
            window_size=20,
            p_switch=0.05,
            n_estimators=10,
            substeps=1,
            event_tolerance=30,
        )
        self.assertEqual(row["hurst"], 0.35)
        self.assertEqual(row["sigma_scale"], 1.0)
        for representation in ("statistics", "signature", "combined"):
            self.assertIn(representation + "_test_ba", row)
            self.assertIn(representation + "_test_false_switches_per_1000", row)
        self.assertEqual(default_output_dir("scale_only", 0.2, 1.0), Path("results/main_grid/fbm_scale_only_h0.2"))
        self.assertEqual(
            default_output_dir("matched_marginal", 0.35, 2.0),
            Path("results/main_grid/fbm_matched_marginal_h0.35_sigma2"),
        )

    def test_variance_audit_compares_rough_setting_with_heston_reference(self):
        audit = stationary_variance_audit(
            scenario="scale_only", hurst=0.2, sigma_scale=1.0, seed=5, n_steps=1500, substeps=1
        )
        self.assertEqual(sorted(audit["setting"].unique()), ["heston_reference", "rough"])
        self.assertEqual(len(audit), 6)
        rough = audit[audit["setting"] == "rough"].set_index("regime")
        reference = audit[audit["setting"] == "heston_reference"].set_index("regime")
        # Same parameter family under both settings.
        pd.testing.assert_series_equal(rough["sigma"], reference["sigma"])
        # Rough noise is antipersistent and shrinks the variance dispersion.
        self.assertTrue((rough["variance_increment_lag1_autocorr"] < -0.2).all())
        self.assertTrue((rough["variance_sd_ratio_to_heston"] < 0.8).all())
        self.assertTrue(reference["variance_sd_ratio_to_heston"].isna().all())


FROZEN_RESULT_DIRS = (
    "results/experiment_a_scale_only",
    "results/experiment_a_scale_only_detector",
    "results/experiment_b_diagnostics",
    "results/experiment_b_window_stability",
    "results/beyond_marginal_benchmark",
)


class MainGridConfigurationTests(unittest.TestCase):
    def test_runners_default_to_ten_validation_paths_and_fresh_grid_directories(self):
        self.assertEqual(DEFAULT_N_PATHS, (10, 10, 5))
        runners = (
            (experiment_a_monte_carlo, ["prog"]),
            (experiment_b_monte_carlo, ["prog"]),
            (experiment_b_window_stability, ["prog"]),
            (experiment_fbm_monte_carlo, ["prog", "--scenario", "scale_only", "--hurst", "0.2"]),
        )
        for module, argv in runners:
            with mock.patch.object(sys, "argv", argv):
                args = module.parse_args()
            self.assertEqual(
                (args.train_paths, args.validation_paths, args.test_paths), DEFAULT_N_PATHS
            )
            self.assertEqual(args.signature_kind, "log")
            output_dir = (
                args.output_dir
                if args.output_dir is not None
                else experiment_fbm_monte_carlo.default_output_dir(
                    args.scenario, args.hurst, args.sigma_scale
                )
            )
            self.assertEqual(output_dir.parts[:2], ("results", "main_grid"))
            self.assertNotIn(str(output_dir), FROZEN_RESULT_DIRS)
        for directory in FROZEN_RESULT_DIRS[:4]:
            config = pd.read_json(PROJECT_ROOT / directory / "config.json", typ="series")
            self.assertEqual(list(config["n_paths"]), [10, 3, 5])

    def test_fresh_runs_never_overwrite_existing_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            guard_output_directory(output_dir, resume=False)  # nothing there yet
            output_dir.mkdir()
            (output_dir / "config.json").write_text("{}")
            guard_output_directory(output_dir, resume=False)  # config alone is fine
            (output_dir / "replications.csv").write_text("replication\n1\n")
            with self.assertRaisesRegex(FileExistsError, "resume"):
                guard_output_directory(output_dir, resume=False)
            guard_output_directory(output_dir, resume=True)


if __name__ == "__main__":
    unittest.main()
