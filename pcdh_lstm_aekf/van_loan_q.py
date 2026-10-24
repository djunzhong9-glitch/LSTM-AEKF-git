import numpy as np
from scipy.linalg import expm

#van_loan_q 的任务是把连续加速度噪声 PSD 转成离散状态协方差 Q。
def van_loan_q(a_matrix: np.ndarray, q_accel_psd: np.ndarray, dt: float) -> np.ndarray:
    """Discretize continuous acceleration PSD into a physical state covariance."""
    #gamma是加速度噪声
    gamma = np.vstack((np.zeros((3, 3)), np.eye(3)))
    #噪声扩散矩阵
    diffusion = gamma @ np.diag(q_accel_psd) @ gamma.T
    #van Loan 方法
    block = np.zeros((12, 12), dtype=float)
    block[:6, :6] = a_matrix
    block[:6, 6:] = diffusion
    block[6:, 6:] = -a_matrix.T
    #把连续噪声精确离散化
    exponential = expm(block * dt)
    phi = exponential[:6, :6]
    q_discrete = exponential[:6, 6:] @ phi.T
    #正定化，对称化。
    q_discrete = 0.5 * (q_discrete + q_discrete.T)
    #修正负特征值
    eigen_min = float(np.linalg.eigvalsh(q_discrete).min())
    if eigen_min < 0.0:
        q_discrete += np.eye(6) * (-eigen_min + 1e-15)
    
    return q_discrete

