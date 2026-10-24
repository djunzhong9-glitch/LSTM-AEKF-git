from collections import deque
import time

import numpy as np

from .build_features import build_features
from .constrained_q import (
    assemble_q_from_basis,
    build_q_basis,
    covariance_matching_sample,
    estimate_psd_from_discrete_q,
    measurement_noise_sample,
    max_abs_lag1_correlation,
    rate_limited_log_update,
)
from .cw_matrices import cw_matrices
from .ekf_step import ekf_step


def precompute_lstm_outputs(model, features: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate one complete fixed-feature sequence once for repeated filter variants."""

    acceleration, log_scales, _, _ = model.forward(
        np.asarray(features, dtype=float)[None, :, :]
    )
    return {
        "acceleration": acceleration[0],
        "log_scales": log_scales[0],
    }


def run_adaptive_ekf(
    sequence: dict[str, np.ndarray],
    model,
    config,
    *,
    use_acceleration_head: bool = True,
    use_q_head: bool = True,
    use_r_head: bool = True,
    use_qr_separation: bool = True,
    use_shadow_fixed_features: bool = True,
    use_q_cm_correction: bool = True,
    use_r_cm_correction: bool = True,
    use_nis_feedback: bool = True,
    use_r_fast_channel: bool = True,
    use_q_slow_constraint: bool = True,
    propagate_acceleration_uncertainty: bool = True,
    precomputed_shadow_features: np.ndarray | None = None,
    precomputed_model_outputs: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray | float]:
    """Run LSTM-assisted AEKF with constrained slow Q update and fast R update."""

    # 调用 CW 矩阵函数，得到连续系统矩阵 A、离散状态转移矩阵 Phi 和加速度输入矩阵 G。
    a_matrix, phi_nominal, g_nominal = cw_matrices(float(sequence["mean_motion"]), config.dt)
    steps = sequence["truth"].shape[0]
    if precomputed_shadow_features is not None:
        precomputed_shadow_features = np.asarray(
            precomputed_shadow_features, dtype=float
        )
        if precomputed_shadow_features.shape != (steps, 15):
            raise ValueError("precomputed_shadow_features must have shape (steps, 15)")
        if not use_shadow_fixed_features:
            raise ValueError(
                "Precomputed shadow features require use_shadow_fixed_features=True"
            )
    if precomputed_model_outputs is not None:
        if np.asarray(precomputed_model_outputs["acceleration"]).shape != (steps, 3):
            raise ValueError("Precomputed acceleration output must have shape (steps, 3)")
        if np.asarray(precomputed_model_outputs["log_scales"]).shape != (steps, 6):
            raise ValueError("Precomputed log-scale output must have shape (steps, 6)")
    estimates = np.zeros((steps, 6), dtype=float)
    covariances = np.zeros((steps, 6, 6), dtype=float)
    features = np.zeros((steps, 15), dtype=float)
    nis = np.full(steps, np.nan)
    nees = np.full(steps, np.nan)
    dof = np.zeros(steps, dtype=int)
    acceleration_estimate = np.zeros((steps, 3), dtype=float)
    acceleration_covariance = np.zeros((steps, 3, 3), dtype=float)
    q_estimate = np.zeros((steps, 3), dtype=float)
    r_estimate = np.zeros((steps, 3), dtype=float)

    state = sequence["initial_estimate"].copy()
    covariance = np.diag(config.p0_diag)
    acceleration_current = np.zeros(3, dtype=float)
    acceleration_covariance_current = np.zeros((3, 3), dtype=float)
    q_current = config.q0_psd.copy()
    r_current = config.r0_var.copy()

    # 中文说明：训练集中的 LSTM 输入来自 Fixed EKF。
    # 如果在线阶段改用 Adaptive EKF 自己的特征，会产生训练/推理分布不一致，
    # 容易导致 Q、R 和加速度头互相补偿。这里额外维护一个影子 Fixed EKF，
    # 它不参与最终导航输出，只负责给 LSTM 生成与训练阶段一致的诊断特征。
    shadow_state = sequence["initial_estimate"].copy()
    shadow_covariance = np.diag(config.p0_diag)

    # LSTM 的隐藏状态用于保留时间序列记忆。
    hidden = None
    # NIS 指数平均用于调节 R；Q 不直接使用该反馈，防止 R 突变时 Q 被误抬高。
    nis_ema = 1.0
    smoothed_q_log_scales = np.zeros(3, dtype=float)
    smoothed_r_log_scales = np.zeros(3, dtype=float)
    r_lower = config.r0_var * np.power(10.0, config.log_scale_min)
    r_upper = config.r0_var * np.power(10.0, config.log_scale_max)
    r_cm_window: deque[np.ndarray] = deque(maxlen=config.r_cm_window)

    # 构建动态约束 Q 的 Van Loan 基矩阵，并给连续 PSD 设置合理上下界。
    q_basis = build_q_basis(a_matrix, config.dt)
    q_lower = config.q0_psd * np.power(10.0, config.q_log_scale_min)
    q_upper = config.q0_psd * np.power(10.0, config.q_log_scale_max)
    q_cm_window: deque[np.ndarray] = deque(maxlen=config.q_cm_window)
    innovation_window: deque[np.ndarray] = deque(maxlen=config.q_cm_window)
    mask_window: deque[np.ndarray] = deque(maxlen=config.q_cm_window)

    start = time.perf_counter()

    for k in range(steps):
        phi = np.eye(6) if k == 0 else phi_nominal
        g_matrix = np.zeros((6, 3)) if k == 0 else g_nominal
        covariance_without_q = phi @ covariance @ phi.T
        covariance_without_q = 0.5 * (covariance_without_q + covariance_without_q.T)
        q_discrete = (
            np.zeros((6, 6))
            if k == 0
            else assemble_q_from_basis(q_current, q_basis)
        )
        if (
            k > 0
            and use_acceleration_head
            and propagate_acceleration_uncertainty
        ):
            q_discrete = (
                q_discrete
                + g_matrix @ acceleration_covariance_current @ g_matrix.T
            )
            q_discrete = 0.5 * (q_discrete + q_discrete.T)

        result = ekf_step(
            state, covariance, sequence["measurements"][k], sequence["mask"][k],
            phi, g_matrix, q_discrete, r_current, acceleration_current,
        )
        if use_shadow_fixed_features:
            if precomputed_shadow_features is None:
                shadow_q_discrete = (
                    np.zeros((6, 6))
                    if k == 0
                    else assemble_q_from_basis(config.q0_psd, q_basis)
                )
                shadow_result = ekf_step(
                    shadow_state,
                    shadow_covariance,
                    sequence["measurements"][k],
                    sequence["mask"][k],
                    phi,
                    g_matrix,
                    shadow_q_discrete,
                    config.r0_var,
                    np.zeros(3, dtype=float),
                )
                shadow_state = shadow_result["state"]
                shadow_covariance = shadow_result["covariance"]
                feature_result = shadow_result
            else:
                feature_result = None
        else:
            feature_result = result
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

        # 构造 LSTM 输入特征：使用影子 Fixed EKF 的创新、预测协方差和量测雅可比。
        if precomputed_shadow_features is not None:
            features[k] = precomputed_shadow_features[k]
        else:
            features[k] = build_features(
                feature_result["innovation"],
                feature_result["covariance_pred"],
                feature_result["jacobian"],
                config.r0_var,
                config.p_ref,
                sequence["mask"][k],
            )

        # LSTM 继续负责未建模加速度补偿；R 头作为量测噪声变化的快速先验。
        if precomputed_model_outputs is None:
            acceleration_next, raw_log_scales, hidden = model.step(features[k], hidden)
        else:
            acceleration_next = np.asarray(
                precomputed_model_outputs["acceleration"][k], dtype=float
            )
            raw_log_scales = np.asarray(
                precomputed_model_outputs["log_scales"][k], dtype=float
            )
        if not use_acceleration_head:
            acceleration_next = np.zeros(3, dtype=float)
        # 中文说明：Q 的主估计采用 LSTM 的监督输出，因为训练标签中已经包含真实 Q 缩放。
        # 协方差匹配只做小修正，不能再主导 Q，否则单条轨迹会把状态修正量误当成过程噪声。
        raw_q_log_scales = np.clip(raw_log_scales[:3], config.q_log_scale_min, config.q_log_scale_max)
        if not use_q_head:
            raw_q_log_scales = np.zeros(3, dtype=float)
        raw_r_log_scales = raw_log_scales[3:] if use_r_head else np.zeros(3, dtype=float)
        if use_q_slow_constraint:
            smoothed_q_log_scales = (
                config.q_lstm_ema_beta * smoothed_q_log_scales
                + (1.0 - config.q_lstm_ema_beta) * raw_q_log_scales
            )
        else:
            smoothed_q_log_scales = raw_q_log_scales.copy()
        smoothed_r_log_scales = (
            config.r_ema_beta * smoothed_r_log_scales
            + (1.0 - config.r_ema_beta) * raw_r_log_scales
        )

        if result["dof"] > 0 and np.isfinite(result["nis"]):
            instant_ratio = max(float(result["nis"]) / int(result["dof"]), 1e-6)
            nis_ema = config.ema_beta * nis_ema + (1.0 - config.ema_beta) * instant_ratio

        # NIS 反馈只加到 R：创新突然变大时，优先解释为量测噪声变化，而不是同时抬高 Q。
        feedback_r = (
            config.nis_feedback_gain * np.log10(max(nis_ema, 1e-6))
            if use_nis_feedback and (use_r_fast_channel or not use_qr_separation)
            else 0.0
        )
        target_r_log_scales = np.clip(
            smoothed_r_log_scales + feedback_r,
            config.log_scale_min,
            config.log_scale_max,
        )
        target_r = config.r0_var * np.power(10.0, target_r_log_scales)

        if result["dof"] > 0 and np.isfinite(result["nis"]):
            r_cm_window.append(
                measurement_noise_sample(
                    result["innovation"],
                    result["covariance_pred"],
                    result["jacobian"],
                    sequence["mask"][k],
                    r_lower,
                    r_upper,
                )
            )
            q_cm_window.append(
                covariance_matching_sample(
                    covariance,
                    covariance_without_q,
                    state,
                    result["state_pred"],
                )
            )
            innovation_window.append(result["innovation"].copy())
            mask_window.append(sequence["mask"][k].copy())

        # 中文说明：R 额外使用创新协方差匹配。这样 R 突变时可以更快上升，
        # 减少后续 Q 为量测异常“背锅”的机会。
        if (
            use_r_fast_channel
            and use_r_cm_correction
            and len(r_cm_window) >= max(4, config.r_cm_window // 2)
        ):
            r_stack = np.stack(tuple(r_cm_window), axis=0)
            valid_counts = np.sum(np.isfinite(r_stack), axis=0)
            with np.errstate(invalid="ignore"):
                r_cm = np.nanmean(r_stack, axis=0)
            usable = valid_counts >= max(3, config.r_cm_window // 4)
            r_candidate = target_r.copy()
            r_candidate[usable] = (
                (1.0 - config.r_cm_weight) * target_r[usable]
                + config.r_cm_weight * r_cm[usable]
            )
        else:
            r_candidate = target_r
        r_candidate = np.clip(r_candidate, r_lower, r_upper)
        r_blended = (1.0 - config.r_update_alpha) * r_current + config.r_update_alpha * r_candidate
        r_current = rate_limited_log_update(r_current, r_blended, config.r0_var, config.r_max_log_step)
        r_current = np.clip(r_current, r_lower, r_upper)

        # Q 采用“LSTM 主估计 + CM 小幅校正 + Van Loan 动态约束 + 慢更新”。
        # 创新有连续相关时，CM 才能对 LSTM 的 Q 目标做有限校正；
        # 创新近似白噪声时，不让 CM 影响 Q，避免把 R 或瞬时量测异常折算成过程噪声。
        feedback_q = 0.0 if use_qr_separation else feedback_r
        q_target_log_scales = smoothed_q_log_scales + feedback_q
        if (
            use_q_slow_constraint
            and use_q_cm_correction
            and len(q_cm_window) == config.q_cm_window
        ):
            q_cm_estimate = np.mean(np.stack(tuple(q_cm_window), axis=0), axis=0)
            q_candidate = estimate_psd_from_discrete_q(q_cm_estimate, q_basis, q_lower, q_upper)
            innovation_corr = max_abs_lag1_correlation(innovation_window, mask_window)
            has_model_error_evidence = (
                (not use_qr_separation)
                or innovation_corr >= config.q_innovation_corr_threshold
            )
            if has_model_error_evidence:
                q_cm_log_scales = np.log10(np.maximum(q_candidate, 1e-300) / config.q0_psd)
                bounded_cm_log_scales = np.clip(
                    q_cm_log_scales,
                    q_target_log_scales - config.q_cm_log_correction_limit,
                    q_target_log_scales + config.q_cm_log_correction_limit,
                )
                q_target_log_scales = (
                    (1.0 - config.q_cm_correction_weight) * q_target_log_scales
                    + config.q_cm_correction_weight * bounded_cm_log_scales
                )

        q_target_log_scales = np.clip(
            q_target_log_scales,
            config.q_log_scale_min,
            config.q_log_scale_max,
        )
        q_target = config.q0_psd * np.power(10.0, q_target_log_scales)
        if use_q_slow_constraint:
            q_blended = (
                (1.0 - config.q_lstm_update_alpha) * q_current
                + config.q_lstm_update_alpha * q_target
            )
            q_current = rate_limited_log_update(
                q_current, q_blended, config.q0_psd, config.q_max_log_step
            )
        else:
            q_current = q_target
        q_current = np.clip(q_current, q_lower, q_upper)
        if not use_q_head and use_qr_separation:
            q_current = config.q0_psd.copy()
        if not use_r_head:
            r_current = config.r0_var.copy()

        acceleration_estimate[k] = acceleration_next
        acceleration_covariance[k] = (
            model.acceleration_error_covariance
            if use_acceleration_head and propagate_acceleration_uncertainty
            else np.zeros((3, 3), dtype=float)
        )
        acceleration_current = acceleration_next
        acceleration_covariance_current = acceleration_covariance[k]

    elapsed = time.perf_counter() - start
    return {
        "estimate": estimates,
        "covariance": covariances,
        "features": features,
        "nis": nis,
        "nees": nees,
        "dof": dof,
        "acceleration_estimate": acceleration_estimate,
        "acceleration_error_covariance": acceleration_covariance,
        "propagate_acceleration_uncertainty": bool(
            use_acceleration_head and propagate_acceleration_uncertainty
        ),
        "q_estimate": q_estimate,
        "r_estimate": r_estimate,
        "runtime_ms_per_step": 1000.0 * elapsed / steps,
    }
