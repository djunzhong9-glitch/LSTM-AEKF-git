from collections import deque
import time

import numpy as np

from .build_dataset import build_gm_residual_features
from .constrained_q import measurement_noise_sample, rate_limited_log_update
from .cw_matrices import cw_matrices
from .run_fixed_ekf import run_fixed_ekf
from .run_gm_ekf import _gm_measurement_step, gm_9d_discretization, run_gm_9d_ekf


def run_gm_residual_lstm_ekf(
    sequence: dict[str, object],
    model,
    config,
    *,
    use_residual_head: bool = True,
    use_lstm_r_head: bool = False,
    use_r_cm_correction: bool = True,
    use_nis_feedback: bool = False,
    propagate_residual_uncertainty: bool = False,
    precomputed_shadow_features: np.ndarray | None = None,
    precomputed_model_outputs: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray | float | bool]:
    """Run a GM-9D EKF whose remaining acceleration mismatch is corrected by LSTM.

    A dedicated residual model predicts the acceleration left after the GM state.
    The primary hybrid keeps covariance adaptation in the physical GM/R-CM filter;
    the optional LSTM R head is retained only for ablation. Legacy total-acceleration
    models remain supported for reproducibility. No true acceleration, true Q, or
    true R is used by the filter.
    """

    if model is None:
        raise ValueError("A trained LSTM model is required for GM residual filtering")

    a_matrix, _, g_nominal = cw_matrices(float(sequence["mean_motion"]), config.dt)
    transition_nominal, process_covariance_nominal = gm_9d_discretization(
        float(sequence["mean_motion"]),
        config.dt,
        config.q0_psd,
        config.gm_tau_s,
        config.gm_drive_psd,
    )
    truth = np.asarray(sequence["truth"])
    measurements = np.asarray(sequence["measurements"])
    masks = np.asarray(sequence["mask"])
    steps = truth.shape[0]

    if precomputed_shadow_features is not None:
        precomputed_shadow_features = np.asarray(precomputed_shadow_features, dtype=float)
        if precomputed_shadow_features.shape != (steps, model.input_size):
            raise ValueError(
                "precomputed_shadow_features must have shape "
                f"(steps, {model.input_size})"
            )
    else:
        if model.output_mode in {"gm_residual", "gm_total_acceleration"}:
            shadow_gm = run_gm_9d_ekf(
                sequence, config, adapt_r=use_r_cm_correction
            )
            precomputed_shadow_features = build_gm_residual_features(
                shadow_gm, config
            )
        else:
            precomputed_shadow_features = run_fixed_ekf(sequence, config)["features"]
    if precomputed_model_outputs is not None:
        if np.asarray(precomputed_model_outputs["acceleration"]).shape != (steps, 3):
            raise ValueError("Precomputed acceleration output must have shape (steps, 3)")
        if np.asarray(precomputed_model_outputs["log_scales"]).shape != (steps, 6):
            raise ValueError("Precomputed log-scale output must have shape (steps, 6)")

    estimates = np.zeros((steps, 6), dtype=float)
    covariances = np.zeros((steps, 6, 6), dtype=float)
    features = np.zeros((steps, model.input_size), dtype=float)
    nis = np.full(steps, np.nan)
    nees = np.full(steps, np.nan)
    dof = np.zeros(steps, dtype=int)
    gm_acceleration_estimate = np.zeros((steps, 3), dtype=float)
    lstm_acceleration_estimate = np.zeros((steps, 3), dtype=float)
    residual_acceleration_estimate = np.zeros((steps, 3), dtype=float)
    r_estimate = np.zeros((steps, 3), dtype=float)

    state = np.zeros(9, dtype=float)
    state[:6] = np.asarray(sequence["initial_estimate"])
    covariance = np.zeros((9, 9), dtype=float)
    covariance[:6, :6] = np.diag(config.p0_diag)
    stationary_acceleration_variance = config.gm_drive_psd * config.gm_tau_s / 2.0
    covariance[6:, 6:] = np.eye(3) * stationary_acceleration_variance

    residual_current = np.zeros(3, dtype=float)
    residual_covariance_current = np.zeros((3, 3), dtype=float)
    r_current = config.r0_var.copy()
    r_lower = config.r0_var * np.power(10.0, config.log_scale_min)
    r_upper = config.r0_var * np.power(10.0, config.log_scale_max)
    smoothed_r_log_scales = np.zeros(3, dtype=float)
    nis_ema = 1.0
    r_cm_window: deque[np.ndarray] = deque(maxlen=config.r_cm_window)
    hidden = None
    start = time.perf_counter()

    for k in range(steps):
        transition = np.eye(9) if k == 0 else transition_nominal
        process_covariance = (
            np.zeros((9, 9), dtype=float) if k == 0 else process_covariance_nominal
        )
        g_matrix = np.zeros((6, 3), dtype=float) if k == 0 else g_nominal
        residual_covariance = (
            residual_covariance_current
            if k > 0 and use_residual_head and propagate_residual_uncertainty
            else None
        )
        result = _gm_measurement_step(
            state,
            covariance,
            measurements[k],
            masks[k],
            transition,
            process_covariance,
            r_current,
            residual_current if use_residual_head else np.zeros(3, dtype=float),
            g_matrix,
            residual_covariance,
        )
        state = np.asarray(result["state"])
        covariance = np.asarray(result["covariance"])
        estimates[k] = state[:6]
        covariances[k] = covariance[:6, :6]
        nis[k] = float(result["nis"])
        dof[k] = int(result["dof"])
        gm_acceleration_estimate[k] = state[6:]
        residual_acceleration_estimate[k] = residual_current
        r_estimate[k] = r_current
        error = truth[k] - state[:6]
        nees[k] = float(error @ np.linalg.solve(covariance[:6, :6], error))

        features[k] = precomputed_shadow_features[k]
        if precomputed_model_outputs is None:
            learned_acceleration_next, raw_log_scales, hidden = model.step(
                features[k], hidden
            )
        else:
            learned_acceleration_next = np.asarray(
                precomputed_model_outputs["acceleration"][k], dtype=float
            )
            raw_log_scales = np.asarray(
                precomputed_model_outputs["log_scales"][k], dtype=float
            )

        if model.output_mode == "gm_residual":
            residual_next = (
                learned_acceleration_next * model.residual_injection_gain
            )
            lstm_acceleration_estimate[k] = state[6:] + residual_next
        elif model.output_mode == "gm_total_acceleration":
            shadow_gm_acceleration = (
                features[k, -9:-6] * model.a_max
            )
            residual_next = (
                learned_acceleration_next - shadow_gm_acceleration
            ) * model.residual_injection_gain
            lstm_acceleration_estimate[k] = learned_acceleration_next
        else:
            lstm_acceleration_estimate[k] = learned_acceleration_next
            residual_next = (
                learned_acceleration_next - state[6:]
            ) * model.residual_injection_gain
        if not use_residual_head:
            residual_next = np.zeros(3, dtype=float)
        residual_current = np.clip(residual_next, -config.a_max, config.a_max)
        residual_covariance_current = (
            model.acceleration_error_covariance
            if use_residual_head and propagate_residual_uncertainty
            else np.zeros((3, 3), dtype=float)
        )

        if int(result["dof"]) > 0 and np.isfinite(result["nis"]):
            instant_ratio = max(float(result["nis"]) / int(result["dof"]), 1e-6)
            nis_ema = config.ema_beta * nis_ema + (1.0 - config.ema_beta) * instant_ratio
            r_cm_window.append(
                measurement_noise_sample(
                    np.asarray(result["innovation"]),
                    np.asarray(result["covariance_pred"]),
                    np.asarray(result["jacobian"]),
                    masks[k],
                    r_lower,
                    r_upper,
                )
            )

        raw_r_log_scales = raw_log_scales[3:] if use_lstm_r_head else np.zeros(3)
        smoothed_r_log_scales = (
            config.r_ema_beta * smoothed_r_log_scales
            + (1.0 - config.r_ema_beta) * raw_r_log_scales
        )
        feedback_r = (
            config.nis_feedback_gain * np.log10(max(nis_ema, 1e-6))
            if use_nis_feedback
            else 0.0
        )
        target_r = config.r0_var * np.power(
            10.0,
            np.clip(
                smoothed_r_log_scales + feedback_r,
                config.log_scale_min,
                config.log_scale_max,
            ),
        )
        if use_r_cm_correction and len(r_cm_window) >= max(4, config.r_cm_window // 2):
            r_stack = np.stack(tuple(r_cm_window), axis=0)
            valid_counts = np.sum(np.isfinite(r_stack), axis=0)
            with np.errstate(invalid="ignore"):
                r_cm = np.nanmean(r_stack, axis=0)
            usable = valid_counts >= max(3, config.r_cm_window // 4)
            target_r[usable] = (
                (1.0 - config.r_cm_weight) * target_r[usable]
                + config.r_cm_weight * r_cm[usable]
            )
        r_candidate = np.clip(target_r, r_lower, r_upper)
        r_blended = (1.0 - config.r_update_alpha) * r_current + (
            config.r_update_alpha * r_candidate
        )
        r_current = rate_limited_log_update(
            r_current,
            r_blended,
            config.r0_var,
            config.r_max_log_step,
        )
        r_current = np.clip(r_current, r_lower, r_upper)

    elapsed = time.perf_counter() - start
    return {
        "estimate": estimates,
        "covariance": covariances,
        "features": features,
        "nis": nis,
        "nees": nees,
        "dof": dof,
        "acceleration_estimate": gm_acceleration_estimate + residual_acceleration_estimate,
        "gm_acceleration_estimate": gm_acceleration_estimate,
        "lstm_acceleration_estimate": lstm_acceleration_estimate,
        "residual_acceleration_estimate": residual_acceleration_estimate,
        "r_estimate": r_estimate,
        "runtime_ms_per_step": 1000.0 * elapsed / steps,
        "uses_true_acceleration": False,
        "uses_true_qr": False,
    }
