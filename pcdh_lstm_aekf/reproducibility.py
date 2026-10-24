from dataclasses import asdict
import json
import platform
from pathlib import Path
import sys

import matplotlib
import numpy as np
import scipy

from .build_dataset import TRAINING_SCENARIOS
from .unmodeled_acceleration import (
    DISTURBANCE_RANDOMIZATION_VERSION,
    IN_DISTRIBUTION_FAMILY,
    LEARNABLE_RESIDUAL_STRESS_FAMILY,
    STRUCTURED_NONSTATIONARY_FAMILY,
)


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def build_reproducibility_record(
    config,
    model,
    training_history: dict[str, object],
    *,
    mode: str,
    scenario: str,
    ood_scenario: str | None,
    monte_carlo_runs: int,
    gm_tuning: dict[str, object],
) -> dict:
    """Build a machine-readable record of data, model, training, and filter settings."""

    all_sequence_seeds = [
        int(config.seed + 101 * index) for index in range(config.train_sequences)
    ]
    training_indices = training_history["training_indices"]
    validation_indices = training_history["validation_indices"]
    q_lower = config.q0_psd * np.power(10.0, config.q_log_scale_min)
    q_upper = config.q0_psd * np.power(10.0, config.q_log_scale_max)
    r_lower = config.r0_var * np.power(10.0, config.log_scale_min)
    r_upper = config.r0_var * np.power(10.0, config.log_scale_max)

    return _json_ready(
        {
            "experiment": {
                "mode": mode,
                "test_scenario": scenario,
                "monte_carlo_runs": monte_carlo_runs,
                "base_seed": config.seed,
                "monte_carlo_seed_offset": 1_000_000,
                "test_seed_formula": "base_seed + 1_000_000 + zero_based_run_index",
                "time_step_s": config.dt,
                "sequence_length_steps": config.steps,
                "sequence_duration_s": (config.steps - 1) * config.dt,
            },
            "data": {
                "training_scenarios_cycle": TRAINING_SCENARIOS,
                "truth_process_noise": {
                    "continuous_model": "white acceleration with PSD diag(q_a)",
                    "discrete_sampling": (
                        "one six-dimensional state increment per interval sampled "
                        "from N(0, Qd), with Qd computed by Van Loan"
                    ),
                    "zero_order_hold_acceleration_sampling": False,
                },
                "rng_stream_policy": (
                    "SeedSequence(seed).spawn(5): structure, acceleration, initial, "
                    "process, measurement"
                ),
                "acceleration_domain_randomization": {
                    "version": DISTURBANCE_RANDOMIZATION_VERSION,
                    "legacy_training_and_id_family": IN_DISTRIBUTION_FAMILY,
                    "legacy_structured_family": STRUCTURED_NONSTATIONARY_FAMILY,
                    "structured_id_family": LEARNABLE_RESIDUAL_STRESS_FAMILY,
                    "structured_id_mechanisms": (
                        "state gates, scheduled pulse-decay events, chirped "
                        "sinusoids, short micro-pulses, and shadow/drag base terms"
                    ),
                    "nominal_amplitude_mps2": config.acceleration_nominal_amplitude,
                    "amplitude_scale_range": config.acceleration_amplitude_scale_range,
                    "nominal_angular_frequency_radps": (
                        config.acceleration_nominal_angular_frequency
                    ),
                    "frequency_scale_range": config.acceleration_frequency_scale_range,
                    "single_step_time_fraction_range": (
                        config.acceleration_step_fraction_range
                    ),
                    "nominal_step_abs_mps2": config.acceleration_nominal_step_abs,
                    "single_step_scale_range": config.acceleration_step_scale_range,
                    "trajectory_parameters_saved_in_matrix": True,
                },
                "training_sequence_count_total": config.train_sequences,
                "training_sequence_seeds_all": all_sequence_seeds,
                "split_level": "complete sequences",
                "split_method": "seeded permutation using base_seed",
                "validation_fraction_requested": config.validation_fraction,
                "training_indices_zero_based": training_indices,
                "validation_indices_zero_based": validation_indices,
                "training_sequence_seeds": [all_sequence_seeds[index] for index in training_indices],
                "validation_sequence_seeds": [
                    all_sequence_seeds[index] for index in validation_indices
                ],
                "training_sequence_count": training_history["training_sequences"],
                "validation_sequence_count": training_history["validation_sequences"],
            },
            "lstm": {
                "implementation": "single-layer NumPy LSTM with full-sequence BPTT",
                "input_dimension": model.input_size,
                "input_feature_order": [
                    "normalized innovation: range, elevation, azimuth (3)",
                    "log10 normalized innovation energy (3)",
                    "log10 predicted state covariance diagonal / p_ref (6)",
                    "measurement availability mask (3)",
                    "GM-residual model only: progress, orbital phase harmonics, and short-period clock (6)",
                    "GM-residual model only: normalized GM navigation state (6)",
                    "GM-residual model only: normalized kinematic acceleration residual and smoothed copy (6)",
                    "GM-residual model only: normalized GM acceleration, first difference, and uncertainty (9)",
                ],
                "feature_normalization": {
                    "innovation": "nu / sqrt(diag(H P_pred H^T + R0))",
                    "innovation_energy": "log10(normalized_innovation^2 + 1e-8)",
                    "covariance": "log10(max(diag(P_pred) / p_ref, 1e-12))",
                    "dataset_standardization": "none; normalization is physics/reference based",
                },
                "sequence_length_steps": config.steps,
                "truncated_bptt": False,
                "hidden_dimension": model.hidden_size,
                "output_dimension": 9,
                "outputs": ["acceleration (3)", "log10 Q scale (3)", "log10 R scale (3)"],
                "hidden_and_cell_initialization": "zeros at the start of every sequence",
                "output_activations": {
                    "acceleration": "a_max elementwise multiplied by tanh(raw acceleration)",
                    "covariance_log10_scales": (
                        "max_log_scale multiplied by tanh(raw covariance scales)"
                    ),
                },
                "output_bounds": {
                    "acceleration_abs_max_mps2": model.a_max,
                    "covariance_log10_scale_abs_max": model.max_log_scale,
                },
            },
            "training": {
                "epochs_requested": config.epochs,
                "batch_size_sequences": config.batch_size,
                "optimizer": "Adam",
                "learning_rate": config.learning_rate,
                "adam_beta1": model.adam_beta1,
                "adam_beta2": model.adam_beta2,
                "adam_epsilon": model.adam_epsilon,
                "gradient_clip_global_norm": config.gradient_clip_norm,
                "loss": "L_a + 0.7 L_QR + 0.02 L_smooth",
                "loss_terms": {
                    "L_a": "MSE of acceleration error normalized by a_max",
                    "L_QR": "MSE of six predicted versus target log10 Q/R scales",
                    "L_smooth": "MSE of first differences of six predicted log10 scales",
                },
                "loss_weights": model.loss_weights,
                "model_selection": "minimum validation total loss; best parameters restored",
                "training_seed": training_history.get("training_seed", config.seed),
                "split_seed": training_history.get("split_seed", config.seed),
                "best_epoch": training_history["best_epoch"],
                "best_validation_loss": training_history["best_validation_loss"],
                "acceleration_error_covariance_m2_s4": (
                    training_history.get("acceleration_error_covariance_m2_s4")
                ),
                "acceleration_error_covariance_source": (
                    training_history.get("acceleration_error_covariance_source")
                ),
                "training_loss_by_epoch": training_history["training_loss"],
                "validation_loss_by_epoch": training_history["validation_loss"],
            },
            "noise_bounds": {
                "q0_acceleration_psd": config.q0_psd,
                "q_log10_scale_bounds": [config.q_log_scale_min, config.q_log_scale_max],
                "q_psd_lower": q_lower,
                "q_psd_upper": q_upper,
                "r0_variance": config.r0_var,
                "r_log10_scale_bounds": [config.log_scale_min, config.log_scale_max],
                "r_variance_lower": r_lower,
                "r_variance_upper": r_upper,
                "r_standard_deviation_lower": np.sqrt(r_lower),
                "r_standard_deviation_upper": np.sqrt(r_upper),
            },
            "adaptive_parameters": {
                "nis_ema_beta": config.ema_beta,
                "nis_feedback_gain": config.nis_feedback_gain,
                "nis_feedback_floor_epsilon": 1.0e-6,
                "acceleration_uncertainty_propagation": (
                    "Qd_total = Qd + G Sigma_a_validation G^T"
                ),
                "r_lstm_ema_beta": config.r_ema_beta,
                "r_cm_window": config.r_cm_window,
                "r_cm_minimum_window": max(4, config.r_cm_window // 2),
                "r_update_alpha": config.r_update_alpha,
                "r_cm_weight": config.r_cm_weight,
                "r_max_log10_step": config.r_max_log_step,
                "q_lstm_ema_beta": config.q_lstm_ema_beta,
                "q_cm_window": config.q_cm_window,
                "q_lstm_update_alpha": config.q_lstm_update_alpha,
                "q_cm_correction_weight": config.q_cm_correction_weight,
                "q_cm_log10_correction_limit": config.q_cm_log_correction_limit,
                "q_max_log10_step": config.q_max_log_step,
                "q_trigger": "max absolute lag-1 innovation correlation >= threshold",
                "q_innovation_corr_threshold": config.q_innovation_corr_threshold,
                "baseline_innovation_cm_window": config.baseline_cm_window,
                "baseline_innovation_cm_alpha": config.baseline_cm_alpha,
                "baseline_sage_husa_forgetting": config.baseline_sage_husa_forgetting,
                "baseline_fading_bounds": [
                    config.baseline_fading_min,
                    config.baseline_fading_max,
                ],
                "baseline_coupled_nis_ema_beta": config.baseline_coupled_ema_beta,
                "baseline_coupled_update_gain": config.baseline_coupled_gain,
                "baseline_coupled_max_log10_step": config.baseline_coupled_max_log_step,
                "gm_9d_model": {
                    "state": "[r(3), v(3), a_u(3)]",
                    "acceleration_dynamics": "a_u_dot = -a_u/tau + w_a",
                    "nominal_white_acceleration_psd": config.q0_psd,
                    "selected_tau_s": config.gm_tau_s,
                    "selected_drive_psd_m2_s5": config.gm_drive_psd,
                    "initial_acceleration_covariance": "stationary q_drive * tau / 2",
                    "r_cm_variant": (
                        "same window, alpha, bounds, and log-rate limit as Innovation-CM"
                    ),
                    "validation_tuning": gm_tuning,
                },
            },
            "initialization": {
                "initial_covariance_diagonal": config.p0_diag,
                "feature_covariance_reference": config.p_ref,
                "nominal_initial_state": config.initial_state,
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "matplotlib": matplotlib.__version__,
            },
            "raw_config": asdict(config),
        }
    )


def save_reproducibility_record(record: dict, output_path: Path) -> None:
    output_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
