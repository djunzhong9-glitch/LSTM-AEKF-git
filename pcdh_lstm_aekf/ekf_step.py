import numpy as np

from .measurement_function import measurement_function
from .measurement_jacobian import measurement_jacobian
from .wrap_angle import wrap_angle


def ekf_step(
    state: np.ndarray,
    covariance: np.ndarray,
    measurement: np.ndarray,
    mask: np.ndarray,
    phi: np.ndarray,#转移矩阵
    g_matrix: np.ndarray,#扰动输入矩阵
    q_discrete: np.ndarray,#就是Q_k
    r_variance: np.ndarray,#就是R_k
    acceleration: np.ndarray,#外部扰动
) -> dict[str, np.ndarray | float | int]:
    """Perform one Joseph-form EKF prediction and masked measurement update."""
    #g_matrix @ acceleration是外部加速度影响
    state_pred = phi @ state + g_matrix @ acceleration
    #covariance_pred是预测协方差矩阵，是因为运动模型不是完美的，存在过程噪声。
    covariance_pred = phi @ covariance @ phi.T + q_discrete
    #保证协方差对称。
    covariance_pred = 0.5 * (covariance_pred + covariance_pred.T)
    #计算量测雅可比
    h_full = measurement_jacobian(state_pred)
    #初始化创新新息
    innovation_full = np.zeros(3, dtype=float)
    #通过mask，判断有效测量。
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
    #预测的量测
    predicted_measurement = measurement_function(state_pred)
    #计算创新新息
    innovation_full[valid] = measurement[valid] - predicted_measurement[valid]
    if mask[2]:
        #将方位角归一化
        innovation_full[2] = wrap_angle(innovation_full[2])
    h = h_full[valid]
    r = np.diag(r_variance[valid])
    innovation = innovation_full[valid]
    #计算创新新息协方差
    s_matrix = h @ covariance_pred @ h.T + r
    #计算卡尔曼增益-> K = P_pred * H^T * S^-1
    gain = np.linalg.solve(s_matrix, h @ covariance_pred).T
    #更新状态
    state_post = state_pred + gain @ innovation
    #创建单位矩阵
    identity = np.eye(6)
    #计算后验协方差矩阵
    #residual_factor = I-KH
    residual_factor = identity - gain @ h
    #covariance_post=(I-KH)P(I-KH)^T+KRK^T
    covariance_post = residual_factor @ covariance_pred @ residual_factor.T + gain @ r @ gain.T
    #保证协方差对称。
    covariance_post = 0.5 * (covariance_post + covariance_post.T)
    #计算NIS指标，检测创新新息是否符合预期。例如NIS突然很大，则说明测量异常。
    nis = float(innovation @ np.linalg.solve(s_matrix, innovation))
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

