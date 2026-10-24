import numpy as np


def measurement_jacobian(state: np.ndarray) -> np.ndarray:
    """Evaluate the range/elevation/azimuth Jacobian at a state estimate."""

    x, y, z = state[:3]
    rho = float(np.linalg.norm(state[:3]))
    s = float(np.hypot(x, y))
    if rho < 1e-9 or s < 1e-9:
        raise ValueError("Degenerate range/angle Jacobian geometry.")
    h = np.zeros((3, 6), dtype=float)
    h[0, :3] = [x / rho, y / rho, z / rho]
    h[1, :3] = [-x * z / (rho**2 * s), -y * z / (rho**2 * s), s / rho**2]
    h[2, :3] = [-y / s**2, x / s**2, 0.0]
    return h

