import numpy as np
from scipy.linalg import expm


def cw_matrices(mean_motion: float, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the continuous CW matrix and exact discrete Phi/G matrices."""

    n = mean_motion
    #创建连续矩阵A
    a = np.zeros((6, 6), dtype=float)
    a[:3, 3:] = np.eye(3)
    a[3, 0] = 3.0 * n**2
    a[3, 4] = 2.0 * n
    a[4, 3] = -2.0 * n
    a[5, 2] = -n**2
    #gamma表示外部加速度如何进入状态。
    gamma = np.vstack((np.zeros((3, 3)), np.eye(3)))
    augmented = np.zeros((9, 9), dtype=float)
    augmented[:6, :6] = a
    augmented[:6, 6:] = gamma
    #矩阵指数
    discrete = expm(augmented * dt)
    #返回连续矩阵A，离散矩阵Phi和G
    return a, discrete[:6, :6], discrete[:6, 6:]

