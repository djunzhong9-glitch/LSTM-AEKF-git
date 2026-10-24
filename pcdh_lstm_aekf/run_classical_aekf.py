from collections import deque
import time

import numpy as np

from .build_features import build_features
from .constrained_q import (
    assemble_q_from_basis,
    build_q_basis,
    measurement_noise_sample,
    rate_limited_log_update,
)
from .cw_matrices import cw_matrices
from .ekf_step import ekf_step


def _measurement_noise_update(
    sample_window: deque[np.ndarray],
    current_r: np.ndarray,
    lower_r: np.ndarray,
    upper_r: np.ndarray,
    alpha: float,
) -> np.ndarray:
    stack = np.stack(tuple(sample_window), axis=0)
    valid_counts = np.sum(np.isfinite(stack), axis=0)
    with np.errstate(invalid="ignore"):
        estimate = np.nanmean(stack, axis=0)
    usable = valid_counts >= max(3, len(sample_window) // 4)
    candidate = current_r.copy()
    candidate[usable] = estimate[usable]
    return np.clip((1.0 - alpha) * current_r + alpha * candidate, lower_r, upper_r)


def _run_classical_adaptive_ekf(
    sequence: dict[str, np.ndarray],
    config,
    method: str,
) -> dict[str, np.ndarray | float]:
    """Run one traditional adaptive EKF using the common EKF core and bounds."""

    if method not in {"innovation_cm", "sage_husa_fading", "coupled_nis"}:
        raise ValueError(f"Unknown classical adaptive method: {method}")

    a_matrix, phi_nominal, g_nominal = cw_matrices(float(sequence["mean_motion"]), config.dt)
    q_basis = build_q_basis(a_matrix, config.dt)
    steps = sequence["truth"].shape[0]
    estimates = np.zeros((steps, 6), dtype=float)
    covariances = np.zeros((steps, 6, 6), dtype=float)
    features = np.zeros((steps, 15), dtype=float)
    nis = np.full(steps, np.nan)
    nees = np.full(steps, np.nan)
    dof = np.zeros(steps, dtype=int)
    q_estimate = np.zeros((steps, 3), dtype=float)
    r_estimate = np.zeros((steps, 3), dtype=float)

    state = sequence["initial_estimate"].copy()
    covariance = np.diag(config.p0_diag)
    q_current = config.q0_psd.copy()
    r_current = config.r0_var.copy()
    q_lower = config.q0_psd * np.power(10.0, config.q_log_scale_min)
    q_upper = config.q0_psd * np.power(10.0, config.q_log_scale_max)
    r_lower = config.r0_var * np.power(10.0, config.log_scale_min)
    r_upper = config.r0_var * np.power(10.0, config.log_scale_max)
    innovation_window: deque[np.ndarray] = deque(maxlen=config.baseline_cm_window)
    nis_ema = 1.0
    fading_factor = 1.0
    shared_log_scale = 0.0
    start = time.perf_counter()

    for k in range(steps):
        phi = np.eye(6) if k == 0 else phi_nominal
        g_matrix = np.zeros((6, 3)) if k == 0 else g_nominal
        q_discrete = (
            np.zeros((6, 6))
            if k == 0
            else assemble_q_from_basis(q_current, q_basis)
        )
        if method == "sage_husa_fading" and k > 0 and fading_factor > 1.0:
            predicted_without_q = phi @ covariance @ phi.T
            q_discrete = q_discrete + (fading_factor - 1.0) * predicted_without_q

        result = ekf_step(
            state,
            covariance,
            sequence["measurements"][k],
            sequence["mask"][k],
            phi,
            g_matrix,
            q_discrete,
            r_current,
            np.zeros(3, dtype=float),
        )
        state = result["state"]
        covariance = result["covariance"]
        estimates[k] = state
        covariances[k] = covariance
        nis[k] = result["nis"]
        dof[k] = result["dof"]
        q_estimate[k] = q_current
        r_estimate[k] = r_current
        error = sequence["truth"][k] - state
        nees[k] = float(error @ np.linalg.solve(covariance, error))
        features[k] = build_features(
            result["innovation"],
            result["covariance_pred"],
            result["jacobian"],
            config.r0_var,
            config.p_ref,
            sequence["mask"][k],
        )

        if result["dof"] <= 0 or not np.isfinite(result["nis"]):
            continue

        instant_ratio = max(float(result["nis"]) / int(result["dof"]), 1e-6)
        if method == "coupled_nis":
            nis_ema = (
                config.baseline_coupled_ema_beta * nis_ema
                + (1.0 - config.baseline_coupled_ema_beta) * instant_ratio
            )
            delta = np.clip(
                config.baseline_coupled_gain * np.log10(max(nis_ema, 1e-6)),
                -config.baseline_coupled_max_log_step,
                config.baseline_coupled_max_log_step,
            )
            shared_log_scale = float(
                np.clip(
                    shared_log_scale + delta,
                    max(config.q_log_scale_min, config.log_scale_min),
                    min(config.q_log_scale_max, config.log_scale_max),
                )
            )
            # The identical scalar NIS feedback is deliberately applied to both Q and R.
            q_current = np.clip(
                config.q0_psd * np.power(10.0, shared_log_scale), q_lower, q_upper
            )
            r_current = np.clip(
                config.r0_var * np.power(10.0, shared_log_scale), r_lower, r_upper
            )
            continue

        sample = measurement_noise_sample(
            result["innovation"],
            result["covariance_pred"],
            result["jacobian"],
            sequence["mask"][k],
            r_lower,
            r_upper,
        )
        if method == "innovation_cm":
            innovation_window.append(sample)
            if len(innovation_window) >= max(4, config.baseline_cm_window // 2):
                candidate = _measurement_noise_update(
                    innovation_window,
                    r_current,
                    r_lower,
                    r_upper,
                    config.baseline_cm_alpha,
                )
                r_current = rate_limited_log_update(
                    r_current, candidate, config.r0_var, config.r_max_log_step
                )
                r_current = np.clip(r_current, r_lower, r_upper)
        else:
            valid = np.isfinite(sample)
            forgetting = config.baseline_sage_husa_forgetting
            r_current[valid] = (
                forgetting * r_current[valid] + (1.0 - forgetting) * sample[valid]
            )
            r_current = np.clip(r_current, r_lower, r_upper)
            nis_ema = config.ema_beta * nis_ema + (1.0 - config.ema_beta) * instant_ratio
            fading_factor = float(
                np.clip(
                    max(1.0, nis_ema),
                    config.baseline_fading_min,
                    config.baseline_fading_max,
                )
            )

    elapsed = time.perf_counter() - start
    return {
        "estimate": estimates,
        "covariance": covariances,
        "features": features,
        "nis": nis,
        "nees": nees,
        "dof": dof,
        "q_estimate": q_estimate,
        "r_estimate": r_estimate,
        "runtime_ms_per_step": 1000.0 * elapsed / steps,
    }


def run_innovation_cm_aekf(sequence, config):
    """Innovation-covariance-matching AEKF with fixed nominal Q and adaptive R."""

    return _run_classical_adaptive_ekf(sequence, config, "innovation_cm")


def run_sage_husa_fading_aekf(sequence, config):
    """Sage-Husa R estimation combined with NIS-driven fading covariance inflation."""

    return _run_classical_adaptive_ekf(sequence, config, "sage_husa_fading")


def run_coupled_nis_aekf(sequence, config):
    """Baseline that applies one shared NIS feedback scale to both Q and R."""

    return _run_classical_adaptive_ekf(sequence, config, "coupled_nis")
