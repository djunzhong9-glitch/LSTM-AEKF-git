import numpy as np


def measurement_function(state: np.ndarray) -> np.ndarray:
    """Map Cartesian relative state to range, elevation and azimuth."""

    x, y, z = state[:3]
    rho = float(np.linalg.norm(state[:3]))
    horizontal = float(np.hypot(x, y))
    #防止为0
    if rho < 1e-9 or horizontal < 1e-9:
        raise ValueError("Degenerate range/angle measurement geometry.")
    #返回：距离，仰角，方位角
    return np.array([rho, np.arctan2(z, horizontal), np.arctan2(y, x)], dtype=float)

