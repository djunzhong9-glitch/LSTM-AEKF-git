from __future__ import annotations

from collections import deque

import numpy as np

from .van_loan_q import van_loan_q


def build_q_basis(a_matrix: np.ndarray, dt: float) -> np.ndarray:
    """Build Van-Loan basis matrices for unit acceleration PSD on x/y/z."""

    # 中文说明：Van Loan 离散化对连续加速度 PSD 是线性的。
    # 分别令 x/y/z 三个方向 PSD 为 1，就能得到三个物理基矩阵。
    # 后续任意物理可实现的离散 Q 都可以写成这三个基矩阵的非负线性组合。
    basis = np.zeros((3, 6, 6), dtype=float)
    for axis in range(3):
        unit_psd = np.zeros(3, dtype=float)
        unit_psd[axis] = 1.0
        basis[axis] = van_loan_q(a_matrix, unit_psd, dt)
    return basis


def assemble_q_from_basis(q_accel_psd: np.ndarray, q_basis: np.ndarray) -> np.ndarray:
    """Assemble an exact Van Loan covariance from precomputed unit-PSD bases."""

    q_discrete = np.tensordot(np.asarray(q_accel_psd, dtype=float), q_basis, axes=(0, 0))
    return 0.5 * (q_discrete + q_discrete.T)


def covariance_matching_sample(
    covariance_post: np.ndarray,
    covariance_without_q: np.ndarray,
    state_post: np.ndarray,
    state_pred: np.ndarray,
) -> np.ndarray:
    """Return one covariance-matching sample of the discrete process covariance."""

    # 中文说明：协方差匹配用后验协方差、无过程噪声预测协方差、
    # 以及本步状态修正量估计“当前窗口需要多少过程噪声”。
    correction = state_post - state_pred
    sample = covariance_post - covariance_without_q + np.outer(correction, correction)
    return 0.5 * (sample + sample.T)


def estimate_psd_from_discrete_q(
    q_discrete_estimate: np.ndarray,
    q_basis: np.ndarray,
    lower_psd: np.ndarray,
    upper_psd: np.ndarray,
) -> np.ndarray:
    """Fit a physical acceleration PSD to a covariance-matching Q estimate."""

    # 中文说明：普通协方差匹配可能得到任意形状的 Q。
    # 这里把它投影到 Van Loan 动力学允许的三轴加速度 PSD 空间，保证 Q 可解释、非负、有边界。
    lower_triangle = np.tril_indices(6)
    design = np.stack([q_basis[axis][lower_triangle] for axis in range(3)], axis=1)
    target = q_discrete_estimate[lower_triangle]

    estimate, *_ = np.linalg.lstsq(design, target, rcond=None)
    estimate = np.nan_to_num(estimate, nan=0.0, posinf=upper_psd, neginf=0.0)
    return np.clip(estimate, lower_psd, upper_psd)


def measurement_noise_sample(
    innovation: np.ndarray,
    covariance_pred: np.ndarray,
    jacobian: np.ndarray,
    mask: np.ndarray,
    lower_var: np.ndarray,
    upper_var: np.ndarray,
) -> np.ndarray:
    """Return one diagonal R covariance-matching sample."""

    # 中文说明：量测创新协方差满足 S = H P_pred H^T + R。
    # 因此单步上可用 innovation^2 - H P_pred H^T 得到 R 的粗略样本；
    # 再经过滑动窗口、上下界和慢速融合，作为 R 突变时的快速证据。
    sample = np.full(3, np.nan, dtype=float)
    predicted_measurement_var = np.einsum("ij,jk,ik->i", jacobian, covariance_pred, jacobian)
    valid = np.flatnonzero(mask)
    if valid.size:
        raw = np.square(innovation[valid]) - predicted_measurement_var[valid]
        sample[valid] = np.clip(raw, lower_var[valid], upper_var[valid])
    return sample


def max_abs_lag1_correlation(
    innovation_window: deque[np.ndarray],
    mask_window: deque[np.ndarray],
) -> float:
    """Estimate whether innovations contain persistent model-error correlation."""

    if len(innovation_window) < 4:
        return 0.0

    innovations = np.asarray(innovation_window, dtype=float)
    masks = np.asarray(mask_window, dtype=bool)
    correlations: list[float] = []
    for channel in range(innovations.shape[1]):
        valid = masks[:-1, channel] & masks[1:, channel]
        if valid.sum() < 4:
            continue
        previous_values = innovations[:-1, channel][valid]
        next_values = innovations[1:, channel][valid]
        if np.std(previous_values) <= 1e-12 or np.std(next_values) <= 1e-12:
            continue
        correlations.append(abs(float(np.corrcoef(previous_values, next_values)[0, 1])))

    return float(max(correlations)) if correlations else 0.0


def rate_limited_log_update(
    current_values: np.ndarray,
    target_values: np.ndarray,
    reference_values: np.ndarray,
    max_log_step: float,
) -> np.ndarray:
    """Move positive values toward a target with a per-step log10 rate limit."""

    current_log = np.log10(np.maximum(current_values, 1e-300) / reference_values)
    target_log = np.log10(np.maximum(target_values, 1e-300) / reference_values)
    limited_log = np.clip(target_log, current_log - max_log_step, current_log + max_log_step)
    return reference_values * np.power(10.0, limited_log)
