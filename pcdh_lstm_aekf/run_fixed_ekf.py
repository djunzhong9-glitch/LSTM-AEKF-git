import time

import numpy as np

from .build_features import build_features
from .cw_matrices import cw_matrices
from .ekf_step import ekf_step
from .van_loan_q import van_loan_q


def run_fixed_ekf(sequence: dict[str, np.ndarray], config) -> dict[str, np.ndarray | float]:
    """Run the fixed nominal EKF and retain features for supervised training."""
    #调用CW矩阵函数，获取连续矩阵A，离散矩阵Phi和G
    a_matrix, phi_nominal, g_nominal = cw_matrices(float(sequence["mean_motion"]), config.dt)
    q_nominal = van_loan_q(a_matrix, config.q0_psd, config.dt)
    #读取仿真长度，就是步数。
    steps = sequence["truth"].shape[0]
    #创建一个保存估计状态的空间
    estimates = np.zeros((steps, 6), dtype=float)
    #保存每一步的协方差
    covariances = np.zeros((steps, 6, 6), dtype=float)
    #保存每一步的特征，就是后面给LSTM模型用的输入特征
    features = np.zeros((steps, 15), dtype=float)
    nis = np.full(steps, np.nan)
    nees = np.full(steps, np.nan)
    #保存有效指标
    dof = np.zeros(steps, dtype=int)
    state = sequence["initial_estimate"].copy()
    covariance = np.diag(config.p0_diag)
    #记录开始时间
    start = time.perf_counter()

    for k in range(steps):
        phi = np.eye(6) if k == 0 else phi_nominal
        g_matrix = np.zeros((6, 3)) if k == 0 else g_nominal
        q_discrete = np.zeros((6, 6)) if k == 0 else q_nominal
        result = ekf_step(
            state, covariance, sequence["measurements"][k], sequence["mask"][k],
            phi, g_matrix, q_discrete, config.r0_var, np.zeros(3),
        )
        state = result["state"]
        covariance = result["covariance"]
        estimates[k] = state
        covariances[k] = covariance
        nis[k] = result["nis"]
        dof[k] = result["dof"]
        error = sequence["truth"][k] - state
        nees[k] = float(error @ np.linalg.solve(covariance, error))
        features[k] = build_features(
            result["innovation"], result["covariance_pred"], result["jacobian"],
            config.r0_var, config.p_ref, sequence["mask"][k],
        )

    elapsed = time.perf_counter() - start
    return {
        "estimate": estimates,
        "covariance": covariances,
        "features": features,
        "nis": nis,
        "nees": nees,
        "dof": dof,
        "runtime_ms_per_step": 1000.0 * elapsed / steps,
    }

