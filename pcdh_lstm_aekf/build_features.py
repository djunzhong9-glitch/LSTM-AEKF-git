import numpy as np


def build_features(
    innovation: np.ndarray,
    covariance_pred: np.ndarray,
    jacobian: np.ndarray,
    nominal_r: np.ndarray,
    p_ref: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Build the paper's 15 causal innovation/covariance/availability features."""

    #创新协方差计算
    nominal_s = jacobian @ covariance_pred @ jacobian.T + np.diag(nominal_r)
    #归一化残差
    normalized = innovation / np.sqrt(np.maximum(np.diag(nominal_s), 1e-15))
    normalized = normalized * mask.astype(float)
    #创新能量（对数变换）
    #对标准化残差的平方取常用对数（log10），加小常数防止 log(0)
    #这相当于残差幅度的对数能量，将动态范围压缩，便于模型处理，同时突出较大残差的异常。
    innovation_energy = np.log10(np.square(normalized) + 1e-8)
    #协方差特征（对数变换）
    covariance_feature = np.log10(np.maximum(np.diag(covariance_pred) / p_ref, 1e-12))
    return np.concatenate((normalized, innovation_energy, covariance_feature, mask.astype(float)))

