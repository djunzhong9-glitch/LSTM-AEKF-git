import time

import numpy as np

from .build_features import build_features
from .constrained_q import assemble_q_from_basis, build_q_basis
from .cw_matrices import cw_matrices
from .ekf_step import ekf_step


def run_known_scale_ekf(sequence: dict[str, np.ndarray], config) -> dict[str, np.ndarray | float]:
    """Run an EKF that knows the simulated Q/R scales but not model acceleration."""
    #调用CW矩阵函数，获取连续矩阵A，离散矩阵Phi和G
    a_matrix, phi_nominal, g_nominal = cw_matrices(float(sequence["mean_motion"]), config.dt)
    q_basis = build_q_basis(a_matrix, config.dt)
    steps = sequence["truth"].shape[0]
    estimates = np.zeros((steps, 6), dtype=float)
    covariances = np.zeros((steps, 6, 6), dtype=float)
    features = np.zeros((steps, 15), dtype=float)
    nis = np.full(steps, np.nan)
    nees = np.full(steps, np.nan)
    dof = np.zeros(steps, dtype=int)
    state = sequence["initial_estimate"].copy()
    covariance = np.diag(config.p0_diag)
    start = time.perf_counter()

    for k in range(steps):
        phi = np.eye(6) if k == 0 else phi_nominal
        g_matrix = np.zeros((6, 3)) if k == 0 else g_nominal
        q_index = max(k - 1, 0)
        q_discrete = (
            np.zeros((6, 6)) if k == 0
            else assemble_q_from_basis(sequence["q_psd"][q_index], q_basis)
        )
        result = ekf_step(
            state, covariance, sequence["measurements"][k], sequence["mask"][k],
            phi, g_matrix, q_discrete, sequence["r_var"][k], np.zeros(3),
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
