from __future__ import annotations

import copy
from collections import deque
import math
import time

import numpy as np

from .build_features import build_features
from .constrained_q import (
    max_abs_lag1_correlation,
    measurement_noise_sample,
    rate_limited_log_update,
)
from .measurement_jacobian import measurement_jacobian
from .run_gm_ekf import (
    _ensure_psd,
    _gm_measurement_step,
    _measurement_noise_update,
    gm_9d_discretization,
    run_gm_9d_ekf,
)
from .compute_metrics import compute_segmented_metrics


ADAPTIVE_GM_TUNING_CANDIDATES = (
    {
        "name": "low_activity_smooth",
        "adaptive_gm_tau_min_s": 10.0,
        "adaptive_gm_tau_max_s": 120.0,
        "adaptive_gm_drive_min_factor": 0.40,
        "adaptive_gm_drive_max_factor": 4.0,
        "adaptive_gm_activity_bias": -1.80,
        "adaptive_gm_alpha_ema": 0.90,
        "adaptive_gm_tau_max_log_step": 0.018,
        "adaptive_gm_drive_max_log_step": 0.035,
        "adaptive_gm_use_axiswise_drive": False,
    },
    {
        "name": "medium_activity_smooth",
        "adaptive_gm_tau_min_s": 10.0,
        "adaptive_gm_tau_max_s": 120.0,
        "adaptive_gm_drive_min_factor": 0.40,
        "adaptive_gm_drive_max_factor": 4.0,
        "adaptive_gm_activity_bias": -1.40,
        "adaptive_gm_alpha_ema": 0.90,
        "adaptive_gm_tau_max_log_step": 0.018,
        "adaptive_gm_drive_max_log_step": 0.035,
        "adaptive_gm_use_axiswise_drive": False,
    },
    {
        "name": "weak_drive_adaptation",
        "adaptive_gm_tau_min_s": 15.0,
        "adaptive_gm_tau_max_s": 160.0,
        "adaptive_gm_drive_min_factor": 0.50,
        "adaptive_gm_drive_max_factor": 2.0,
        "adaptive_gm_activity_bias": -1.10,
        "adaptive_gm_alpha_ema": 0.90,
        "adaptive_gm_tau_max_log_step": 0.018,
        "adaptive_gm_drive_max_log_step": 0.020,
        "adaptive_gm_use_axiswise_drive": False,
    },
    {
        "name": "tau_dominant",
        "adaptive_gm_tau_min_s": 15.0,
        "adaptive_gm_tau_max_s": 160.0,
        "adaptive_gm_drive_min_factor": 0.80,
        "adaptive_gm_drive_max_factor": 1.25,
        "adaptive_gm_activity_bias": -1.10,
        "adaptive_gm_alpha_ema": 0.90,
        "adaptive_gm_tau_max_log_step": 0.018,
        "adaptive_gm_drive_max_log_step": 0.010,
        "adaptive_gm_use_axiswise_drive": False,
    },
    {
        "name": "high_smooth_prior",
        "adaptive_gm_tau_min_s": 20.0,
        "adaptive_gm_tau_max_s": 220.0,
        "adaptive_gm_drive_min_factor": 0.25,
        "adaptive_gm_drive_max_factor": 2.5,
        "adaptive_gm_activity_bias": -1.65,
        "adaptive_gm_alpha_ema": 0.92,
        "adaptive_gm_tau_max_log_step": 0.014,
        "adaptive_gm_drive_max_log_step": 0.025,
        "adaptive_gm_use_axiswise_drive": False,
    },
    {
        "name": "axiswise_activity",
        "adaptive_gm_tau_min_s": 10.0,
        "adaptive_gm_tau_max_s": 120.0,
        "adaptive_gm_drive_min_factor": 0.40,
        "adaptive_gm_drive_max_factor": 4.0,
        "adaptive_gm_activity_bias": -1.80,
        "adaptive_gm_alpha_ema": 0.90,
        "adaptive_gm_tau_max_log_step": 0.018,
        "adaptive_gm_drive_max_log_step": 0.035,
        "adaptive_gm_use_axiswise_drive": True,
    },
)


def _as_drive_vector(value: float | np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(3, float(array), dtype=float)
    if array.shape != (3,):
        raise ValueError("drive PSD must be a scalar or a length-3 vector")
    return array.copy()


def _clip_positive(value: float, lower: float, upper: float) -> float:
    return float(np.clip(value, lower, upper))


def _clip_positive_vector(
    value: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    return np.clip(np.asarray(value, dtype=float), lower, upper)


def _log_rate_limit_scalar(
    current_value: float,
    target_value: float,
    reference_value: float,
    max_log_step: float,
) -> float:
    current = np.asarray([current_value], dtype=float)
    target = np.asarray([target_value], dtype=float)
    reference = np.asarray([reference_value], dtype=float)
    return float(rate_limited_log_update(current, target, reference, max_log_step)[0])


def _log_interpolate(lower: float, upper: float, fraction: float) -> float:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return float(np.exp((1.0 - fraction) * np.log(lower) + fraction * np.log(upper)))


def _sigmoid(value: float) -> float:
    value = float(np.clip(value, -50.0, 50.0))
    return float(1.0 / (1.0 + math.exp(-value)))


def _axis_activity_scale(
    acceleration_delta: np.ndarray,
    a_max: np.ndarray,
    floor: float,
    ceiling: float,
) -> np.ndarray:
    normalized = np.abs(acceleration_delta) / np.maximum(a_max, 1e-12)
    if float(np.max(normalized)) <= 1e-12:
        return np.ones(3, dtype=float)
    relative = normalized / max(float(np.mean(normalized)), 1e-12)
    return np.clip(relative, floor, ceiling)


def _transition_from_cache(
    cache: dict[tuple[float, float, tuple[float, float, float]], tuple[np.ndarray, np.ndarray]],
    mean_motion: float,
    dt: float,
    q0_psd: np.ndarray,
    tau_s: float,
    drive_vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    drive_key = tuple(float(round(math.log10(value), 4)) for value in drive_vector)
    key = (float(round(dt, 9)), float(round(math.log10(tau_s), 4)), drive_key)
    if key not in cache:
        cache[key] = gm_9d_discretization(
            mean_motion,
            dt,
            q0_psd,
            tau_s,
            drive_vector,
        )
    return cache[key]


def _innovation_model_score(
    result: dict[str, object],
    r_variance: np.ndarray,
    mask: np.ndarray,
    covariance_penalty: float,
) -> float:
    if int(result["dof"]) <= 0 or not np.isfinite(float(result["nis"])):
        return float("inf")
    valid = np.flatnonzero(mask)
    jacobian = np.asarray(result["jacobian"], dtype=float)[valid]
    covariance_pred = np.asarray(result["covariance_pred"], dtype=float)
    innovation_covariance = (
        jacobian @ covariance_pred @ jacobian.T + np.diag(r_variance[valid])
    )
    sign, logdet = np.linalg.slogdet(innovation_covariance)
    if sign <= 0 or not np.isfinite(logdet):
        return float("inf")
    covariance = np.asarray(result["covariance"], dtype=float)
    trace_ratio = float(np.trace(covariance[:6, :6]) / max(np.sum(r_variance), 1e-30))
    return float(
        float(result["nis"])
        + logdet
        + covariance_penalty * math.log(max(trace_ratio, 1e-30))
    )


def _unique_candidate_pairs(
    tau_current: float,
    drive_current: np.ndarray,
    base_drive: float,
    config,
) -> list[tuple[float, np.ndarray]]:
    candidates = [(tau_current, drive_current.copy())]
    for tau_s, drive_factor in config.adaptive_gm_bank_models:
        candidates.append(
            (
                float(tau_s),
                np.full(3, base_drive * float(drive_factor), dtype=float),
            )
        )
    unique: dict[tuple[float, tuple[float, float, float]], tuple[float, np.ndarray]] = {}
    for tau_s, drive_vector in candidates:
        key = (
            float(round(tau_s, 6)),
            tuple(float(round(value, 15)) for value in drive_vector),
        )
        unique[key] = (tau_s, drive_vector)
    return list(unique.values())


def run_adaptive_gm_ekf(
    sequence: dict[str, object],
    config,
    *,
    adapt_tau: bool = True,
    adapt_drive: bool = True,
    adapt_r: bool = True,
    use_innovation_correlation: bool = True,
    use_nis_activity: bool = True,
    use_acceleration_difference: bool = True,
    use_axiswise_drive: bool | None = None,
    tau_s: float | None = None,
    drive_psd: float | None = None,
) -> dict[str, object]:
    """Run an adaptive 9D Gauss-Markov acceleration-state EKF.

    The baseline GM-9D EKF freezes one acceleration correlation time and one
    drive PSD after validation-set tuning.  This variant keeps the same state
    and measurement model but changes those two GM parameters causally from
    innovation correlation, NIS deviation, and acceleration-state activity.
    """

    mean_motion = float(sequence["mean_motion"])
    base_tau = float(config.gm_tau_s if tau_s is None else tau_s)
    base_drive = float(config.gm_drive_psd if drive_psd is None else drive_psd)
    tau_lower = float(config.adaptive_gm_tau_min_s)
    tau_upper = float(config.adaptive_gm_tau_max_s)
    drive_lower = np.full(3, base_drive * config.adaptive_gm_drive_min_factor)
    drive_upper = np.full(3, base_drive * config.adaptive_gm_drive_max_factor)
    use_axiswise = (
        bool(config.adaptive_gm_use_axiswise_drive)
        if use_axiswise_drive is None
        else bool(use_axiswise_drive)
    )

    truth = np.asarray(sequence["truth"])
    measurements = np.asarray(sequence["measurements"])
    masks = np.asarray(sequence["mask"])
    steps = truth.shape[0]
    estimates = np.zeros((steps, 6), dtype=float)
    acceleration_estimates = np.zeros((steps, 3), dtype=float)
    acceleration_covariances = np.zeros((steps, 3, 3), dtype=float)
    covariances = np.zeros((steps, 6, 6), dtype=float)
    features = np.zeros((steps, 15), dtype=float)
    nis = np.full(steps, np.nan)
    nees = np.full(steps, np.nan)
    dof = np.zeros(steps, dtype=int)
    r_estimate = np.zeros((steps, 3), dtype=float)
    tau_estimate = np.zeros(steps, dtype=float)
    drive_psd_estimate = np.zeros((steps, 3), dtype=float)
    activity_estimate = np.zeros(steps, dtype=float)
    activity_components = np.zeros((steps, 3), dtype=float)
    bank_selected_index = np.full(steps, -1, dtype=int)

    state = np.zeros(9, dtype=float)
    state[:6] = np.asarray(sequence["initial_estimate"])
    covariance = np.zeros((9, 9), dtype=float)
    covariance[:6, :6] = np.diag(config.p0_diag)

    tau_current = _clip_positive(base_tau, tau_lower, tau_upper)
    drive_current = _clip_positive_vector(
        _as_drive_vector(base_drive),
        drive_lower,
        drive_upper,
    )
    covariance[6:, 6:] = np.diag(drive_current * tau_current / 2.0)

    r_current = config.r0_var.copy()
    r_lower = config.r0_var * np.power(10.0, config.log_scale_min)
    r_upper = config.r0_var * np.power(10.0, config.log_scale_max)
    r_sample_window: deque[np.ndarray] = deque(maxlen=config.baseline_cm_window)
    innovation_window: deque[np.ndarray] = deque(maxlen=config.adaptive_gm_window)
    mask_window: deque[np.ndarray] = deque(maxlen=config.adaptive_gm_window)
    transition_cache: dict[
        tuple[float, float, tuple[float, float, float]],
        tuple[np.ndarray, np.ndarray],
    ] = {}

    activity_level = 0.0
    previous_acceleration = state[6:].copy()
    start = time.perf_counter()

    for k in range(steps):
        selected_bank_index = -1
        if k == 0:
            transition = np.eye(9, dtype=float)
            process_covariance = np.zeros((9, 9), dtype=float)
            result = _gm_measurement_step(
                state,
                covariance,
                measurements[k],
                masks[k],
                transition,
                process_covariance,
                r_current,
            )
        elif (adapt_tau or adapt_drive) and bool(config.adaptive_gm_use_likelihood_bank):
            best_score = float("inf")
            best_result = None
            best_tau = tau_current
            best_drive = drive_current.copy()
            for candidate_index, (candidate_tau, candidate_drive) in enumerate(
                _unique_candidate_pairs(tau_current, drive_current, base_drive, config)
            ):
                if not adapt_tau:
                    candidate_tau = base_tau
                if not adapt_drive:
                    candidate_drive = np.full(3, base_drive, dtype=float)
                candidate_tau = _clip_positive(candidate_tau, tau_lower, tau_upper)
                candidate_drive = _clip_positive_vector(
                    candidate_drive,
                    drive_lower,
                    drive_upper,
                )
                transition, process_covariance = _transition_from_cache(
                    transition_cache,
                    mean_motion,
                    config.dt,
                    config.q0_psd,
                    candidate_tau,
                    candidate_drive,
                )
                candidate_result = _gm_measurement_step(
                    state,
                    covariance,
                    measurements[k],
                    masks[k],
                    transition,
                    process_covariance,
                    r_current,
                )
                score = _innovation_model_score(
                    candidate_result,
                    r_current,
                    masks[k],
                    float(config.adaptive_gm_bank_covariance_penalty),
                )
                if score < best_score:
                    best_score = score
                    best_result = candidate_result
                    best_tau = candidate_tau
                    best_drive = candidate_drive
                    selected_bank_index = candidate_index
            if best_result is None:
                transition, process_covariance = _transition_from_cache(
                    transition_cache,
                    mean_motion,
                    config.dt,
                    config.q0_psd,
                    tau_current,
                    drive_current,
                )
                best_result = _gm_measurement_step(
                    state,
                    covariance,
                    measurements[k],
                    masks[k],
                    transition,
                    process_covariance,
                    r_current,
                )
            result = best_result
            tau_current = best_tau
            drive_current = best_drive
        else:
            transition, process_covariance = _transition_from_cache(
                transition_cache,
                mean_motion,
                config.dt,
                config.q0_psd,
                tau_current,
                drive_current,
            )
            result = _gm_measurement_step(
                state,
                covariance,
                measurements[k],
                masks[k],
                transition,
                process_covariance,
                r_current,
            )
        state = np.asarray(result["state"])
        covariance = np.asarray(result["covariance"])

        estimates[k] = state[:6]
        acceleration_estimates[k] = state[6:]
        acceleration_covariances[k] = covariance[6:, 6:]
        covariances[k] = covariance[:6, :6]
        nis[k] = float(result["nis"])
        dof[k] = int(result["dof"])
        r_estimate[k] = r_current
        tau_estimate[k] = tau_current
        drive_psd_estimate[k] = drive_current
        activity_estimate[k] = activity_level
        bank_selected_index[k] = selected_bank_index

        error = truth[k] - state[:6]
        covariance_six = _ensure_psd(covariance[:6, :6])
        nees[k] = float(error @ np.linalg.solve(covariance_six, error))

        jacobian_six = np.asarray(result["jacobian"])[:, :6]
        covariance_pred_six = np.asarray(result["covariance_pred"])[:6, :6]
        features[k] = build_features(
            np.asarray(result["innovation"]),
            covariance_pred_six,
            jacobian_six,
            r_current,
            config.p_ref,
            masks[k],
        )

        valid_measurement = int(result["dof"]) > 0 and np.isfinite(result["nis"])
        if valid_measurement:
            innovation_window.append(np.asarray(result["innovation"], dtype=float).copy())
            mask_window.append(np.asarray(masks[k], dtype=bool).copy())
            if adapt_r:
                sample = measurement_noise_sample(
                    np.asarray(result["innovation"]),
                    np.asarray(result["covariance_pred"]),
                    np.asarray(result["jacobian"]),
                    masks[k],
                    r_lower,
                    r_upper,
                )
                r_sample_window.append(sample)
                if len(r_sample_window) >= max(4, config.baseline_cm_window // 2):
                    candidate = _measurement_noise_update(
                        r_sample_window,
                        r_current,
                        r_lower,
                        r_upper,
                        config.baseline_cm_alpha,
                    )
                    r_current = rate_limited_log_update(
                        r_current,
                        candidate,
                        config.r0_var,
                        config.r_max_log_step,
                    )
                    r_current = np.clip(r_current, r_lower, r_upper)

        corr_activity = (
            max_abs_lag1_correlation(innovation_window, mask_window)
            if use_innovation_correlation
            else 0.0
        )
        nis_ratio = 1.0
        if valid_measurement and int(result["dof"]) > 0:
            nis_ratio = float(result["nis"]) / max(float(result["dof"]), 1.0)
        nis_activity = abs(math.log10(max(nis_ratio, 1e-6))) if use_nis_activity else 0.0
        acceleration_delta = state[6:] - previous_acceleration
        accel_activity = (
            float(
                np.linalg.norm(acceleration_delta / np.maximum(config.a_max, 1e-12))
                / np.sqrt(3.0)
            )
            if use_acceleration_difference
            else 0.0
        )
        activity_components[k] = (
            float(corr_activity),
            float(nis_activity),
            float(accel_activity),
        )

        score = (
            float(config.adaptive_gm_activity_bias)
            + float(config.adaptive_gm_corr_weight) * corr_activity
            + float(config.adaptive_gm_nis_weight) * nis_activity
            + float(config.adaptive_gm_accel_diff_weight) * min(accel_activity, 4.0)
        )
        target_activity = _sigmoid(float(config.adaptive_gm_activity_slope) * score)
        ema = float(config.adaptive_gm_alpha_ema)
        activity_level = float(np.clip(ema * activity_level + (1.0 - ema) * target_activity, 0.0, 1.0))

        tau_target = (
            _log_interpolate(tau_upper, tau_lower, activity_level)
            if adapt_tau
            else base_tau
        )
        drive_scalar_target = (
            _log_interpolate(float(drive_lower[0]), float(drive_upper[0]), activity_level)
            if adapt_drive
            else base_drive
        )
        if adapt_drive and use_axiswise:
            axis_scale = _axis_activity_scale(
                acceleration_delta,
                config.a_max,
                float(config.adaptive_gm_axis_drive_floor),
                float(config.adaptive_gm_axis_drive_ceiling),
            )
            drive_target = np.clip(
                drive_scalar_target * axis_scale,
                drive_lower,
                drive_upper,
            )
        else:
            drive_target = np.full(3, drive_scalar_target, dtype=float)

        tau_current = _log_rate_limit_scalar(
            tau_current,
            _clip_positive(tau_target, tau_lower, tau_upper),
            base_tau,
            float(config.adaptive_gm_tau_max_log_step),
        )
        tau_current = _clip_positive(tau_current, tau_lower, tau_upper)
        drive_current = rate_limited_log_update(
            drive_current,
            _clip_positive_vector(drive_target, drive_lower, drive_upper),
            np.full(3, base_drive, dtype=float),
            float(config.adaptive_gm_drive_max_log_step),
        )
        drive_current = _clip_positive_vector(drive_current, drive_lower, drive_upper)

        if (adapt_tau or adapt_drive) and float(config.adaptive_gm_stationary_reset_weight) > 0.0:
            target_acceleration_covariance = np.diag(drive_current * tau_current / 2.0)
            blend = float(config.adaptive_gm_stationary_reset_weight) * (1.0 - activity_level)
            covariance[6:, 6:] = (
                (1.0 - blend) * covariance[6:, 6:]
                + blend * target_acceleration_covariance
            )
            covariance = _ensure_psd(covariance)

        previous_acceleration = state[6:].copy()

    elapsed = time.perf_counter() - start
    return {
        "estimate": estimates,
        "acceleration_estimate": acceleration_estimates,
        "acceleration_covariance": acceleration_covariances,
        "covariance": covariances,
        "features": features,
        "time": np.asarray(sequence["time"], dtype=float).copy(),
        "mean_motion": mean_motion,
        "nis": nis,
        "nees": nees,
        "dof": dof,
        "r_estimate": r_estimate,
        "tau_estimate": tau_estimate,
        "drive_psd_estimate": drive_psd_estimate,
        "activity_estimate": activity_estimate,
        "activity_components": activity_components,
        "bank_selected_index": bank_selected_index,
        "base_tau_s": base_tau,
        "base_drive_psd": base_drive,
        "adapt_tau": bool(adapt_tau),
        "adapt_drive": bool(adapt_drive),
        "adapt_r": bool(adapt_r),
        "runtime_ms_per_step": 1000.0 * elapsed / steps,
    }


def _candidate_without_name(candidate: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in candidate.items() if key != "name"}


def apply_adaptive_gm_parameters(config, parameters: dict[str, object]) -> None:
    """Apply a frozen Adaptive GM-AEKF parameter set to a config object."""

    for key, value in parameters.items():
        if key == "name":
            continue
        if not hasattr(config, key):
            raise ValueError(f"Unknown Adaptive GM-AEKF config field: {key}")
        setattr(config, key, value)


def tune_adaptive_gm_hyperparameters(
    validation_sequences: list[dict[str, object]],
    config,
    *,
    show_progress: bool = False,
) -> dict[str, object]:
    """Select Adaptive GM-AEKF online adaptation settings on validation data only."""

    if not validation_sequences:
        raise ValueError("At least one validation sequence is required")

    gm_metrics = [
        compute_segmented_metrics(
            sequence,
            run_gm_9d_ekf(sequence, config),
            config,
        )["post_burn_in"]
        for sequence in validation_sequences
    ]
    records = []
    for index, candidate in enumerate(ADAPTIVE_GM_TUNING_CANDIDATES, start=1):
        candidate_config = copy.deepcopy(config)
        apply_adaptive_gm_parameters(candidate_config, _candidate_without_name(candidate))
        position_values = []
        velocity_values = []
        nis_values = []
        nees_values = []
        relative_scores = []
        mean_tau_values = []
        mean_drive_values = []
        mean_activity_values = []
        for sequence, gm_metric in zip(validation_sequences, gm_metrics, strict=True):
            result = run_adaptive_gm_ekf(sequence, candidate_config)
            metrics = compute_segmented_metrics(
                sequence,
                result,
                candidate_config,
            )["post_burn_in"]
            position_values.append(metrics["position_rss_rmse_m"])
            velocity_values.append(metrics["velocity_rss_rmse_mps"])
            nis_values.append(metrics["normalized_nis"])
            nees_values.append(metrics["normalized_nees"])
            mean_tau_values.append(float(np.mean(result["tau_estimate"])))
            mean_drive_values.append(float(np.mean(result["drive_psd_estimate"])))
            mean_activity_values.append(float(np.mean(result["activity_estimate"])))
            relative_scores.append(
                0.55
                * metrics["position_rss_rmse_m"]
                / max(gm_metric["position_rss_rmse_m"], 1e-12)
                + 0.30
                * metrics["velocity_rss_rmse_mps"]
                / max(gm_metric["velocity_rss_rmse_mps"], 1e-12)
                + 0.10
                * abs(math.log(max(metrics["normalized_nis"], 1e-12)))
                + 0.05
                * abs(math.log(max(metrics["normalized_nees"], 1e-12)))
            )
        record = {
            "name": str(candidate["name"]),
            "parameters": _candidate_without_name(candidate),
            "validation_sequence_count": len(validation_sequences),
            "validation_sequence_seeds": [
                int(sequence["seed"]) for sequence in validation_sequences
            ],
            "mean_position_rss_rmse_m": float(np.mean(position_values)),
            "mean_velocity_rss_rmse_mps": float(np.mean(velocity_values)),
            "mean_normalized_nis": float(np.mean(nis_values)),
            "mean_normalized_nees": float(np.mean(nees_values)),
            "mean_tau_s": float(np.mean(mean_tau_values)),
            "mean_drive_psd_m2_s5": float(np.mean(mean_drive_values)),
            "mean_activity_level": float(np.mean(mean_activity_values)),
            "mean_relative_validation_score": float(np.mean(relative_scores)),
        }
        records.append(record)
        if show_progress:
            print(
                "  Adaptive GM validation tuning: "
                f"{index}/{len(ADAPTIVE_GM_TUNING_CANDIDATES)} "
                f"candidates complete ({record['name']})"
            )

    selected = min(records, key=lambda item: item["mean_relative_validation_score"])
    apply_adaptive_gm_parameters(config, selected["parameters"])
    return {
        "selection_data": "LSTM validation trajectories only",
        "selection_metric": (
            "mean validation score = 0.55 position RMSE ratio to fixed GM-9D + "
            "0.30 velocity RMSE ratio to fixed GM-9D + 0.10 |log(NIS/DOF)| + "
            "0.05 |log(NEES/6)|"
        ),
        "candidate_count": len(records),
        "candidate_grid": records,
        "selected_name": selected["name"],
        "selected_parameters": selected["parameters"],
        "selected_score": selected["mean_relative_validation_score"],
        "test_set_used_for_selection": False,
    }
