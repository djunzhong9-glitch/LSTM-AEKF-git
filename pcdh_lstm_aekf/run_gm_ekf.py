from collections import deque
from concurrent.futures import ProcessPoolExecutor
import time

import numpy as np
from scipy.linalg import expm

from .build_features import build_features
from .compute_metrics import compute_metrics
from .constrained_q import measurement_noise_sample, rate_limited_log_update
from .cw_matrices import cw_matrices
from .measurement_function import measurement_function
from .measurement_jacobian import measurement_jacobian
from .run_fixed_ekf import run_fixed_ekf
from .wrap_angle import wrap_angle


_GM_TUNING_CONTEXT: dict[str, object] = {}


def _initialize_gm_tuning_worker(validation_sequences, fixed_metrics, config) -> None:
    global _GM_TUNING_CONTEXT
    _GM_TUNING_CONTEXT = {
        "validation_sequences": validation_sequences,
        "fixed_metrics": fixed_metrics,
        "config": config,
    }


def _evaluate_gm_candidate(
    validation_sequences,
    fixed_metrics,
    config,
    tau_s: float,
    drive_psd: float,
) -> dict[str, float]:
    position_values = []
    velocity_values = []
    relative_scores = []
    for sequence, fixed in zip(validation_sequences, fixed_metrics, strict=True):
        result = run_gm_9d_ekf(
            sequence,
            config,
            tau_s=tau_s,
            drive_psd=drive_psd,
        )
        metrics = compute_metrics(sequence, result)
        position_values.append(metrics["position_rss_rmse_m"])
        velocity_values.append(metrics["velocity_rss_rmse_mps"])
        relative_scores.append(
            0.5
            * (
                metrics["position_rss_rmse_m"]
                / max(fixed["position_rss_rmse_m"], 1e-12)
                + metrics["velocity_rss_rmse_mps"]
                / max(fixed["velocity_rss_rmse_mps"], 1e-12)
            )
        )
    return {
        "tau_s": float(tau_s),
        "drive_psd_m2_s5": float(drive_psd),
        "mean_position_rss_rmse_m": float(np.mean(position_values)),
        "mean_velocity_rss_rmse_mps": float(np.mean(velocity_values)),
        "mean_relative_accuracy_score": float(np.mean(relative_scores)),
    }


def _gm_candidate_worker(payload: tuple[float, float]) -> dict[str, float]:
    return _evaluate_gm_candidate(
        _GM_TUNING_CONTEXT["validation_sequences"],
        _GM_TUNING_CONTEXT["fixed_metrics"],
        _GM_TUNING_CONTEXT["config"],
        payload[0],
        payload[1],
    )


def _ensure_psd(matrix: np.ndarray) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    minimum = float(np.linalg.eigvalsh(matrix).min())
    if minimum < 0.0:
        matrix = matrix + np.eye(matrix.shape[0]) * (-minimum + 1e-15)
    return matrix


def gm_9d_discretization(
    mean_motion: float,
    dt: float,
    q0_psd: np.ndarray,
    tau_s: float,
    drive_psd: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Discretize the CW plus first-order Gauss-Markov acceleration model."""

    drive_array = np.asarray(drive_psd, dtype=float)
    if drive_array.ndim == 0:
        drive_array = np.full(3, float(drive_array), dtype=float)
    if drive_array.shape != (3,):
        raise ValueError("drive_psd must be a positive scalar or a length-3 vector")
    if tau_s <= 0.0 or np.any(drive_array <= 0.0):
        raise ValueError("tau_s and drive_psd must be positive")

    a_matrix, _, _ = cw_matrices(mean_motion, dt)
    gamma = np.vstack((np.zeros((3, 3)), np.eye(3)))
    continuous = np.zeros((9, 9), dtype=float)
    continuous[:6, :6] = a_matrix
    continuous[:6, 6:] = gamma
    continuous[6:, 6:] = -np.eye(3) / tau_s

    diffusion = np.zeros((9, 9), dtype=float)
    diffusion[:6, :6] = gamma @ np.diag(q0_psd) @ gamma.T
    diffusion[6:, 6:] = np.diag(drive_array)

    van_loan = np.zeros((18, 18), dtype=float)
    van_loan[:9, :9] = continuous
    van_loan[:9, 9:] = diffusion
    van_loan[9:, 9:] = -continuous.T
    exponential = expm(van_loan * dt)
    transition = exponential[:9, :9]
    process_covariance = exponential[:9, 9:] @ transition.T
    return transition, _ensure_psd(process_covariance)


def _gm_measurement_step(
    state: np.ndarray,
    covariance: np.ndarray,
    measurement: np.ndarray,
    mask: np.ndarray,
    transition: np.ndarray,
    process_covariance: np.ndarray,
    r_variance: np.ndarray,
    residual_acceleration: np.ndarray | None = None,
    residual_input_matrix: np.ndarray | None = None,
    residual_covariance: np.ndarray | None = None,
) -> dict[str, object]:
    state_pred = transition @ state
    if residual_acceleration is not None:
        if residual_input_matrix is None:
            raise ValueError("residual_input_matrix is required with residual_acceleration")
        state_pred[:6] = state_pred[:6] + residual_input_matrix @ residual_acceleration
    covariance_pred = _ensure_psd(
        transition @ covariance @ transition.T + process_covariance
    )
    if residual_covariance is not None:
        if residual_input_matrix is None:
            raise ValueError("residual_input_matrix is required with residual_covariance")
        covariance_pred[:6, :6] = covariance_pred[:6, :6] + (
            residual_input_matrix @ residual_covariance @ residual_input_matrix.T
        )
        covariance_pred = _ensure_psd(covariance_pred)
    h_six = measurement_jacobian(state_pred[:6])
    h_full = np.zeros((3, 9), dtype=float)
    h_full[:, :6] = h_six
    innovation_full = np.zeros(3, dtype=float)
    valid = np.flatnonzero(mask)

    if valid.size == 0:
        return {
            "state": state_pred,
            "covariance": covariance_pred,
            "state_pred": state_pred,
            "covariance_pred": covariance_pred,
            "innovation": innovation_full,
            "jacobian": h_full,
            "nis": np.nan,
            "dof": 0,
        }

    predicted_measurement = measurement_function(state_pred[:6])
    innovation_full[valid] = measurement[valid] - predicted_measurement[valid]
    if mask[2]:
        innovation_full[2] = wrap_angle(innovation_full[2])
    h = h_full[valid]
    r_matrix = np.diag(r_variance[valid])
    innovation = innovation_full[valid]
    innovation_covariance = h @ covariance_pred @ h.T + r_matrix
    gain = np.linalg.solve(innovation_covariance, h @ covariance_pred).T
    state_post = state_pred + gain @ innovation
    residual_factor = np.eye(9) - gain @ h
    covariance_post = _ensure_psd(
        residual_factor @ covariance_pred @ residual_factor.T
        + gain @ r_matrix @ gain.T
    )
    nis = float(innovation @ np.linalg.solve(innovation_covariance, innovation))
    return {
        "state": state_post,
        "covariance": covariance_post,
        "state_pred": state_pred,
        "covariance_pred": covariance_pred,
        "innovation": innovation_full,
        "jacobian": h_full,
        "nis": nis,
        "dof": int(valid.size),
    }


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


def run_gm_9d_ekf(
    sequence: dict[str, object],
    config,
    *,
    adapt_r: bool = False,
    tau_s: float | None = None,
    drive_psd: float | None = None,
) -> dict[str, object]:
    """Run a 9D Singer/Gauss-Markov acceleration-state augmented EKF."""

    tau_s = float(config.gm_tau_s if tau_s is None else tau_s)
    drive_psd = float(config.gm_drive_psd if drive_psd is None else drive_psd)
    transition_nominal, process_covariance_nominal = gm_9d_discretization(
        float(sequence["mean_motion"]),
        config.dt,
        config.q0_psd,
        tau_s,
        drive_psd,
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

    state = np.zeros(9, dtype=float)
    state[:6] = np.asarray(sequence["initial_estimate"])
    covariance = np.zeros((9, 9), dtype=float)
    covariance[:6, :6] = np.diag(config.p0_diag)
    drive_array = np.asarray(drive_psd, dtype=float)
    if drive_array.ndim == 0:
        drive_array = np.full(3, float(drive_array), dtype=float)
    stationary_acceleration_variance = drive_array * tau_s / 2.0
    covariance[6:, 6:] = np.diag(stationary_acceleration_variance)
    r_current = config.r0_var.copy()
    r_lower = config.r0_var * np.power(10.0, config.log_scale_min)
    r_upper = config.r0_var * np.power(10.0, config.log_scale_max)
    innovation_window: deque[np.ndarray] = deque(maxlen=config.baseline_cm_window)
    start = time.perf_counter()

    for k in range(steps):
        transition = np.eye(9) if k == 0 else transition_nominal
        process_covariance = (
            np.zeros((9, 9), dtype=float) if k == 0 else process_covariance_nominal
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
        error = truth[k] - state[:6]
        nees[k] = float(error @ np.linalg.solve(covariance[:6, :6], error))
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

        if not adapt_r or int(result["dof"]) <= 0 or not np.isfinite(result["nis"]):
            continue
        sample = measurement_noise_sample(
            np.asarray(result["innovation"]),
            np.asarray(result["covariance_pred"]),
            np.asarray(result["jacobian"]),
            masks[k],
            r_lower,
            r_upper,
        )
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
                r_current,
                candidate,
                config.r0_var,
                config.r_max_log_step,
            )
            r_current = np.clip(r_current, r_lower, r_upper)

    elapsed = time.perf_counter() - start
    return {
        "estimate": estimates,
        "acceleration_estimate": acceleration_estimates,
        "acceleration_covariance": acceleration_covariances,
        "covariance": covariances,
        "features": features,
        "time": np.asarray(sequence["time"], dtype=float).copy(),
        "mean_motion": float(sequence["mean_motion"]),
        "nis": nis,
        "nees": nees,
        "dof": dof,
        "r_estimate": r_estimate,
        "tau_s": tau_s,
        "drive_psd": drive_psd,
        "runtime_ms_per_step": 1000.0 * elapsed / steps,
    }


def tune_gm_hyperparameters(
    validation_sequences: list[dict[str, object]],
    config,
    *,
    show_progress: bool = False,
    workers: int = 1,
) -> dict[str, object]:
    """Select GM time constant and drive PSD using validation trajectories only."""

    if not validation_sequences:
        raise ValueError("At least one validation sequence is required for GM tuning")

    fixed_metrics = [
        compute_metrics(sequence, run_fixed_ekf(sequence, config))
        for sequence in validation_sequences
    ]
    payloads = [
        (float(tau_s), float(drive_psd))
        for tau_s in config.gm_tau_candidates_s
        for drive_psd in config.gm_drive_psd_candidates
    ]
    total = len(payloads)
    if workers > 1:
        executor_context = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_gm_tuning_worker,
            initargs=(validation_sequences, fixed_metrics, config),
        )
        result_iterator = executor_context.map(
            _gm_candidate_worker, payloads, chunksize=1
        )
    else:
        executor_context = None
        result_iterator = (
            _evaluate_gm_candidate(
                validation_sequences,
                fixed_metrics,
                config,
                tau_s,
                drive_psd,
            )
            for tau_s, drive_psd in payloads
        )
    try:
        candidates = []
        for candidate_index, candidate in enumerate(result_iterator, start=1):
            candidates.append(candidate)
            if show_progress:
                print(
                    f"  GM validation tuning: {candidate_index}/{total} "
                    "candidates complete"
                )
    finally:
        if executor_context is not None:
            executor_context.shutdown()

    tau_values = tuple(float(value) for value in config.gm_tau_candidates_s)
    drive_values = tuple(float(value) for value in config.gm_drive_psd_candidates)
    raw_best = min(candidates, key=lambda item: item["mean_relative_accuracy_score"])
    best_score = float(raw_best["mean_relative_accuracy_score"])
    plateau_tolerance = 0.01
    near_optimal = [
        candidate
        for candidate in candidates
        if candidate["mean_relative_accuracy_score"]
        <= best_score * (1.0 + plateau_tolerance)
    ]
    interior_near_optimal = [
        candidate
        for candidate in near_optimal
        if tau_values.index(float(candidate["tau_s"])) not in {0, len(tau_values) - 1}
        and drive_values.index(float(candidate["drive_psd_m2_s5"]))
        not in {0, len(drive_values) - 1}
    ]
    selected = min(
        interior_near_optimal if interior_near_optimal else near_optimal,
        key=lambda item: item["mean_relative_accuracy_score"],
    )
    config.gm_tau_s = selected["tau_s"]
    config.gm_drive_psd = selected["drive_psd_m2_s5"]
    selected_tau_index = tau_values.index(float(selected["tau_s"]))
    selected_drive_index = drive_values.index(float(selected["drive_psd_m2_s5"]))
    same_tau_near_drive = sorted(
        candidate["drive_psd_m2_s5"]
        for candidate in near_optimal
        if candidate["tau_s"] == selected["tau_s"]
    )
    selected_on_drive_boundary = selected_drive_index in {0, len(drive_values) - 1}
    drive_plateau = (
        selected_on_drive_boundary
        and len(same_tau_near_drive) >= 2
        and (
            drive_values[-2:] == tuple(same_tau_near_drive[-2:])
            or drive_values[:2] == tuple(same_tau_near_drive[:2])
        )
    )
    selected_on_tau_boundary = selected_tau_index in {0, len(tau_values) - 1}
    grid_status = (
        "interior_near_optimal_plateau"
        if interior_near_optimal
        else "boundary_plateau"
        if len(near_optimal) >= 2
        else "unresolved_boundary_optimum"
    )
    return {
        "selection_data": "LSTM validation trajectories only",
        "selection_metric": (
            "mean over validation trajectories of 0.5 * "
            "(GM position RMSE / Fixed position RMSE + "
            "GM velocity RMSE / Fixed velocity RMSE); candidates within 1% "
            "of the raw optimum form a plateau, then the lowest-score interior "
            "candidate is frozen"
        ),
        "validation_sequence_count": len(validation_sequences),
        "validation_sequence_seeds": [int(sequence["seed"]) for sequence in validation_sequences],
        "validation_sequence_scenarios": [
            str(sequence["scenario"]) for sequence in validation_sequences
        ],
        "candidate_grid": candidates,
        "grid_diagnostics": {
            "selected_tau_index": selected_tau_index,
            "selected_drive_psd_index": selected_drive_index,
            "raw_best_tau_s": raw_best["tau_s"],
            "raw_best_drive_psd_m2_s5": raw_best["drive_psd_m2_s5"],
            "raw_best_relative_accuracy_score": best_score,
            "selected_on_tau_boundary": selected_on_tau_boundary,
            "selected_on_drive_psd_boundary": selected_on_drive_boundary,
            "near_optimal_relative_tolerance": plateau_tolerance,
            "near_optimal_candidate_count": len(near_optimal),
            "interior_near_optimal_candidate_count": len(interior_near_optimal),
            "same_tau_near_optimal_drive_psd_values": same_tau_near_drive,
            "drive_psd_plateau_at_boundary": drive_plateau,
            "drive_grid_status": grid_status,
        },
        "selected_tau_s": selected["tau_s"],
        "selected_drive_psd_m2_s5": selected["drive_psd_m2_s5"],
        "selected_stationary_acceleration_std_mps2": float(
            np.sqrt(selected["drive_psd_m2_s5"] * selected["tau_s"] / 2.0)
        ),
        "test_set_used_for_selection": False,
    }
