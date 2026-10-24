import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pcdh_lstm_aekf.build_dataset import (
    build_gm_residual_dataset,
    build_gm_residual_features,
    build_validation_sequences,
    training_sequence_spec,
)
from pcdh_lstm_aekf.compute_metrics import compute_metrics
from pcdh_lstm_aekf.constrained_q import assemble_q_from_basis, build_q_basis
from pcdh_lstm_aekf.cw_matrices import cw_matrices
from pcdh_lstm_aekf.dual_head_lstm import DualHeadLSTM
from pcdh_lstm_aekf.experiment_matrix import build_method_suite
from pcdh_lstm_aekf.make_config import make_config
from pcdh_lstm_aekf.major_revision_experiments import (
    _factor_task_key,
    _load_factor_checkpoint,
)
from pcdh_lstm_aekf.run_gm_ekf import (
    gm_9d_discretization,
    run_gm_9d_ekf,
    tune_gm_hyperparameters,
)
from pcdh_lstm_aekf.run_adaptive_gm_ekf import run_adaptive_gm_ekf
from pcdh_lstm_aekf.run_gm_residual_lstm_ekf import run_gm_residual_lstm_ekf
from pcdh_lstm_aekf.run_adaptive_ekf import precompute_lstm_outputs, run_adaptive_ekf
from pcdh_lstm_aekf.run_fixed_ekf import run_fixed_ekf
from pcdh_lstm_aekf.simulate_sequence import (
    sample_state_process_increment,
    simulate_sequence,
)


class DisturbanceDomainRandomizationTests(unittest.TestCase):
    def setUp(self):
        self.config = make_config("quick")
        self.config.steps = 120

    def test_profile_is_deterministic_per_seed_and_varies_across_seeds(self):
        first = simulate_sequence(self.config, 1101, "nonlinear")
        repeated = simulate_sequence(self.config, 1101, "nonlinear")
        second = simulate_sequence(self.config, 1102, "nonlinear")

        for key in (
            "amplitudes_mps2",
            "phases_rad",
            "omega_start_radps",
            "omega_rate_radps2",
            "step_times_s",
            "step_biases_mps2",
        ):
            np.testing.assert_allclose(
                first["acceleration_profile"][key],
                repeated["acceleration_profile"][key],
            )
        self.assertFalse(
            np.allclose(
                first["acceleration_profile"]["amplitudes_mps2"],
                second["acceleration_profile"]["amplitudes_mps2"],
            )
        )
        self.assertNotEqual(
            float(first["acceleration_profile"]["step_times_s"][0]),
            float(second["acceleration_profile"]["step_times_s"][0]),
        )

    def test_step_time_scales_with_complete_trajectory_duration(self):
        sequence = simulate_sequence(self.config, 2201, "nonlinear")
        duration = float(sequence["time"][-1])
        step_time = float(sequence["acceleration_profile"]["step_times_s"][0])
        lower, upper = self.config.acceleration_step_fraction_range
        self.assertGreaterEqual(step_time, lower * duration)
        self.assertLessEqual(step_time, upper * duration)

    def test_single_factor_frequency_scan_matches_axis_rms(self):
        sequence = simulate_sequence(
            self.config,
            3351,
            "nonlinear",
            acceleration_variant={
                "factor": "frequency",
                "level": 2.5,
                "rms_match": True,
            },
        )
        profile = sequence["acceleration_profile"]
        np.testing.assert_allclose(
            profile["realized_rms_ratio"], np.ones(3), rtol=2e-12, atol=2e-12
        )
        self.assertEqual(profile["family"], "single_factor_frequency")

    def test_transition_duration_changes_shape_without_changing_rms(self):
        sharp = simulate_sequence(
            self.config,
            3361,
            "nonlinear",
            acceleration_variant={
                "factor": "transition_duration",
                "level": 0.0,
                "rms_match": True,
            },
        )
        smooth = simulate_sequence(
            self.config,
            3361,
            "nonlinear",
            acceleration_variant={
                "factor": "transition_duration",
                "level": 3.0,
                "rms_match": True,
            },
        )
        np.testing.assert_allclose(
            sharp["acceleration_profile"]["realized_rms_mps2"],
            smooth["acceleration_profile"]["realized_rms_mps2"],
            rtol=2e-12,
            atol=2e-12,
        )
        self.assertFalse(np.allclose(sharp["acceleration"], smooth["acceleration"]))

    def test_structured_transition_duration_scan_changes_shape(self):
        sharp = simulate_sequence(
            self.config,
            3362,
            "nonlinear_structured",
            acceleration_variant={
                "factor": "transition_duration",
                "level": 0.0,
                "rms_match": True,
            },
        )
        smooth = simulate_sequence(
            self.config,
            3362,
            "nonlinear_structured",
            acceleration_variant={
                "factor": "transition_duration",
                "level": 15.0,
                "rms_match": True,
            },
        )
        self.assertEqual(
            sharp["acceleration_profile"]["base_family"],
            "learnable_residual_stress",
        )
        self.assertFalse(np.allclose(sharp["acceleration"], smooth["acceleration"]))
        rms_ratio = (
            sharp["acceleration_profile"]["trajectory_acceleration_rms_mps2"]
            / np.maximum(
                smooth["acceleration_profile"]["trajectory_acceleration_rms_mps2"],
                1e-15,
            )
        )
        self.assertTrue(np.all((rms_ratio > 0.65) & (rms_ratio < 1.50)))

    def test_non_gm_friendly_disturbance_scenarios_are_distinct(self):
        base = simulate_sequence(self.config, 3371, "nonlinear")
        scenario_to_family = {
            "nonlinear_pulse": "multi_harmonic_pulse",
            "nonlinear_shadow": "shadow_eclipse_switch",
            "nonlinear_state_drag": "state_dependent_drag_like",
            "nonlinear_structured": "learnable_residual_stress",
            "nonlinear_structured_legacy": "structured_shadow_pulse_drag",
        }
        for scenario, family in scenario_to_family.items():
            with self.subTest(scenario=scenario):
                sequence = simulate_sequence(self.config, 3371, scenario)
                self.assertEqual(sequence["acceleration_profile"]["family"], family)
                self.assertTrue(np.isfinite(sequence["truth"]).all())
                self.assertTrue(np.isfinite(sequence["measurements"]).all())
                self.assertFalse(
                    np.allclose(base["acceleration"], sequence["acceleration"])
                )
                if scenario == "nonlinear_structured":
                    self.assertEqual(
                        len(sequence["acceleration_profile"]["base_components"]),
                        2,
                    )
                    self.assertEqual(
                        sequence["acceleration_profile"]["event_times_s"].size,
                        5,
                    )
                if scenario == "nonlinear_structured_legacy":
                    self.assertEqual(
                        len(sequence["acceleration_profile"]["components"]),
                        3,
                    )


class ContinuousProcessNoiseTests(unittest.TestCase):
    def test_optimized_single_step_matches_full_forward(self):
        config = make_config("quick")
        model = DualHeadLSTM(12, config.a_max, config.log_scale_max, seed=5501)
        rng = np.random.default_rng(5502)
        feature = rng.normal(size=model.input_size)
        hidden = (
            rng.normal(size=(1, model.hidden_size)),
            rng.normal(size=(1, model.hidden_size)),
        )
        expected_acceleration, expected_scales, expected_hidden, _ = model.forward(
            feature.reshape(1, 1, -1), hidden
        )
        acceleration, scales, next_hidden = model.step(feature, hidden)
        np.testing.assert_allclose(acceleration, expected_acceleration[0, 0])
        np.testing.assert_allclose(scales, expected_scales[0, 0])
        np.testing.assert_allclose(next_hidden[0], expected_hidden[0])
        np.testing.assert_allclose(next_hidden[1], expected_hidden[1])

    def test_sampled_state_increment_matches_van_loan_covariance(self):
        config = make_config("quick")
        a_matrix, _, _ = cw_matrices(config.mean_motion, config.dt)
        q_basis = build_q_basis(a_matrix, config.dt)
        target = assemble_q_from_basis(config.q0_psd, q_basis)
        rng = np.random.default_rng(5511)
        samples = np.stack(
            [sample_state_process_increment(rng, target) for _ in range(30000)]
        )
        empirical = np.cov(samples, rowvar=False, ddof=1)
        relative_error = np.linalg.norm(empirical - target) / np.linalg.norm(target)
        self.assertLess(relative_error, 0.04)

    def test_acceleration_error_covariance_enters_state_covariance(self):
        config = make_config("quick")
        config.steps = 40
        sequence = simulate_sequence(config, 5521, "nonlinear")
        model = DualHeadLSTM(8, config.a_max, config.log_scale_max, seed=5522)
        for value in model.parameters.values():
            value.fill(0.0)
        model.acceleration_error_covariance = np.diag([2.0e-7, 1.5e-7, 1.0e-7])
        with_uncertainty = run_adaptive_ekf(
            sequence,
            model,
            config,
            use_q_head=False,
            use_r_head=False,
            use_nis_feedback=False,
            use_q_cm_correction=False,
            use_r_cm_correction=False,
            propagate_acceleration_uncertainty=True,
        )
        deterministic_mean = run_adaptive_ekf(
            sequence,
            model,
            config,
            use_q_head=False,
            use_r_head=False,
            use_nis_feedback=False,
            use_q_cm_correction=False,
            use_r_cm_correction=False,
            propagate_acceleration_uncertainty=False,
        )
        self.assertTrue(with_uncertainty["propagate_acceleration_uncertainty"])
        self.assertFalse(deterministic_mean["propagate_acceleration_uncertainty"])
        self.assertGreater(
            float(np.trace(with_uncertainty["covariance"][-1])),
            float(np.trace(deterministic_mean["covariance"][-1])),
        )

    def test_cached_shadow_features_and_outputs_are_exact(self):
        config = make_config("quick")
        config.steps = 35
        sequence = simulate_sequence(config, 5531, "nonlinear")
        model = DualHeadLSTM(10, config.a_max, config.log_scale_max, seed=5532)
        model.acceleration_error_covariance = np.diag([1.0e-8, 1.1e-8, 0.9e-8])
        uncached = run_adaptive_ekf(sequence, model, config)
        fixed = run_fixed_ekf(sequence, config)
        cached_outputs = precompute_lstm_outputs(model, fixed["features"])
        cached = run_adaptive_ekf(
            sequence,
            model,
            config,
            precomputed_shadow_features=fixed["features"],
            precomputed_model_outputs=cached_outputs,
        )
        for key in ("estimate", "covariance", "features", "q_estimate", "r_estimate"):
            np.testing.assert_allclose(cached[key], uncached[key], rtol=2e-12, atol=2e-12)


class GaussMarkovBaselineTests(unittest.TestCase):
    def setUp(self):
        self.config = make_config("quick")
        self.config.steps = 100

    def test_discrete_process_covariance_is_symmetric_psd(self):
        transition, process_covariance = gm_9d_discretization(
            self.config.mean_motion,
            self.config.dt,
            self.config.q0_psd,
            self.config.gm_tau_s,
            self.config.gm_drive_psd,
        )
        self.assertEqual(transition.shape, (9, 9))
        self.assertEqual(process_covariance.shape, (9, 9))
        np.testing.assert_allclose(process_covariance, process_covariance.T, atol=1e-14)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(process_covariance).min()), -1e-14)

    def test_fixed_r_and_r_cm_variants_are_finite(self):
        sequence = simulate_sequence(self.config, 4401, "nonlinear")
        for adapt_r in (False, True):
            result = run_gm_9d_ekf(sequence, self.config, adapt_r=adapt_r)
            self.assertEqual(result["estimate"].shape, (self.config.steps, 6))
            self.assertEqual(
                result["acceleration_estimate"].shape, (self.config.steps, 3)
            )
            self.assertTrue(np.isfinite(result["estimate"]).all())
            self.assertTrue(np.isfinite(result["covariance"]).all())
            self.assertTrue(compute_metrics(sequence, result)["finite"])

    def test_adaptive_gm_outputs_are_finite_and_bounded(self):
        sequence = simulate_sequence(self.config, 4402, "nonlinear_structured")
        result = run_adaptive_gm_ekf(sequence, self.config)
        self.assertEqual(result["estimate"].shape, (self.config.steps, 6))
        self.assertEqual(result["tau_estimate"].shape, (self.config.steps,))
        self.assertEqual(result["drive_psd_estimate"].shape, (self.config.steps, 3))
        self.assertTrue(np.isfinite(result["estimate"]).all())
        self.assertTrue(np.isfinite(result["covariance"]).all())
        self.assertTrue(np.isfinite(result["tau_estimate"]).all())
        self.assertTrue(np.isfinite(result["drive_psd_estimate"]).all())
        self.assertGreaterEqual(
            float(np.min(result["tau_estimate"])),
            self.config.adaptive_gm_tau_min_s,
        )
        self.assertLessEqual(
            float(np.max(result["tau_estimate"])),
            self.config.adaptive_gm_tau_max_s,
        )
        self.assertTrue(compute_metrics(sequence, result)["finite"])

    def test_disabled_adaptive_gm_reproduces_fixed_gm(self):
        sequence = simulate_sequence(self.config, 4403, "nonlinear_structured")
        gm_result = run_gm_9d_ekf(sequence, self.config, adapt_r=False)
        adaptive_disabled = run_adaptive_gm_ekf(
            sequence,
            self.config,
            adapt_tau=False,
            adapt_drive=False,
            adapt_r=False,
            use_innovation_correlation=False,
            use_nis_activity=False,
            use_acceleration_difference=False,
            use_axiswise_drive=False,
        )
        for key in ("estimate", "covariance", "acceleration_estimate", "nis", "nees"):
            np.testing.assert_allclose(
                adaptive_disabled[key],
                gm_result[key],
                rtol=2e-12,
                atol=2e-12,
            )

    def test_gm_residual_lstm_variant_is_finite_and_not_oracle(self):
        sequence = simulate_sequence(self.config, 4411, "nonlinear_pulse")
        model = DualHeadLSTM(15, self.config.a_max, self.config.log_scale_max, seed=4412)
        for value in model.parameters.values():
            value.fill(0.0)
        model.acceleration_error_covariance = np.diag([1.0e-8, 1.0e-8, 1.0e-8])
        fixed = run_fixed_ekf(sequence, self.config)
        cached_outputs = precompute_lstm_outputs(model, fixed["features"])
        result = run_gm_residual_lstm_ekf(
            sequence,
            model,
            self.config,
            precomputed_shadow_features=fixed["features"],
            precomputed_model_outputs=cached_outputs,
        )
        self.assertEqual(result["estimate"].shape, (self.config.steps, 6))
        self.assertEqual(
            result["residual_acceleration_estimate"].shape,
            (self.config.steps, 3),
        )
        self.assertTrue(np.isfinite(result["estimate"]).all())
        self.assertTrue(np.isfinite(result["covariance"]).all())
        self.assertFalse(result["uses_true_acceleration"])
        self.assertFalse(result["uses_true_qr"])
        self.assertTrue(compute_metrics(sequence, result)["finite"])

    def test_dedicated_residual_features_and_labels_are_causal_and_aligned(self):
        self.config.steps = 45
        self.config.train_sequences = 2
        self.config.gm_lstm_target_mode = "gm_residual"
        features, labels = build_gm_residual_dataset(self.config)
        self.assertEqual(features.shape, (2, 45, 42))
        self.assertEqual(labels.shape, (2, 45, 9))

        seed, scenario = training_sequence_spec(self.config, 0)
        sequence = simulate_sequence(self.config, seed, scenario)
        gm_result = run_gm_9d_ekf(sequence, self.config, adapt_r=True)
        expected = sequence["acceleration"] - gm_result["acceleration_estimate"]
        np.testing.assert_allclose(labels[0, :, :3], expected)
        np.testing.assert_allclose(
            features[0], build_gm_residual_features(gm_result, self.config)
        )

    def test_zero_direct_residual_reproduces_gm_mean(self):
        sequence = simulate_sequence(self.config, 4413, "nonlinear_structured")
        gm_result = run_gm_9d_ekf(sequence, self.config)
        residual_features = build_gm_residual_features(gm_result, self.config)
        model = DualHeadLSTM(
            12,
            self.config.a_max,
            self.config.log_scale_max,
            seed=4414,
            input_size=42,
            output_mode="gm_residual",
        )
        for value in model.parameters.values():
            value.fill(0.0)
        outputs = precompute_lstm_outputs(model, residual_features)
        result = run_gm_residual_lstm_ekf(
            sequence,
            model,
            self.config,
            use_lstm_r_head=False,
            use_r_cm_correction=False,
            use_nis_feedback=False,
            propagate_residual_uncertainty=False,
            precomputed_shadow_features=residual_features,
            precomputed_model_outputs=outputs,
        )
        np.testing.assert_allclose(result["estimate"], gm_result["estimate"])

    def test_method_suite_contains_both_gm_baselines(self):
        methods = build_method_suite(model=None, config=self.config)
        self.assertIn("GM-9D EKF", methods)
        self.assertIn("GM-9D EKF + R-CM", methods)
        self.assertIn("Adaptive GM-AEKF", methods)
        self.assertIn("GM-9D EKF + LSTM residual", methods)

    def test_tuning_uses_only_supplied_validation_sequences(self):
        self.config.gm_tau_candidates_s = (15.0, 45.0)
        self.config.gm_drive_psd_candidates = (2.0e-9, 8.0e-9)
        validation = build_validation_sequences(self.config, [0, 1])
        record = tune_gm_hyperparameters(validation, self.config)
        self.assertEqual(record["validation_sequence_count"], 2)
        self.assertEqual(
            record["validation_sequence_seeds"], [sequence["seed"] for sequence in validation]
        )
        self.assertFalse(record["test_set_used_for_selection"])
        self.assertIn(record["selected_tau_s"], self.config.gm_tau_candidates_s)
        self.assertIn(
            record["selected_drive_psd_m2_s5"],
            self.config.gm_drive_psd_candidates,
        )


class FactorCheckpointTests(unittest.TestCase):
    def test_restores_complete_rows_and_truncates_partial_tail(self):
        payload = ("amplitude_scale", 0.5, 1234)
        record = {
            "task_key": _factor_task_key(payload),
            "factor": payload[0],
            "level": payload[1],
            "test_seed": payload[2],
            "rows": [{"method": "Fixed EKF"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            path.write_text(
                json.dumps(record, separators=(",", ":")) + "\n{partial",
                encoding="utf-8",
            )
            restored = _load_factor_checkpoint(path)
            self.assertEqual(restored[record["task_key"]], record["rows"])
            self.assertEqual(path.read_text(encoding="utf-8").count("\n"), 1)


if __name__ == "__main__":
    unittest.main()
