import numpy as np


def true_dynamics(
    state: np.ndarray,
    mean_motion: float,
    semimajor_axis: float,
    mu: float,
    external_acceleration: np.ndarray,
    #nonlinear 非线性模型，bool类型，True表示使用非线性模型，False表示使用线性模型
    nonlinear: bool,
) -> np.ndarray:
    """Evaluate nonlinear relative two-body or nominal CW dynamics."""

    position = state[:3]
    velocity = state[3:]
    x, y, z = position
    #因为后面的CW模型没有出现vz，所以这里不写。
    vx, vy, _ = velocity
    
    derivative = np.empty(6, dtype=float)
    derivative[:3] = velocity
    if nonlinear:
        radius = np.sqrt((semimajor_axis + x) ** 2 + y**2 + z**2)
        gravity_difference = np.array(
            [
                -mu * (semimajor_axis + x) / radius**3 + mu / semimajor_axis**2,
                -mu * y / radius**3,
                -mu * z / radius**3,
            ]
        )
        rotating_terms = np.array(
            [2.0 * mean_motion * vy + mean_motion**2 * x,
             -2.0 * mean_motion * vx + mean_motion**2 * y,
             0.0]
        )
        derivative[3:] = gravity_difference + rotating_terms + external_acceleration
    else:
        derivative[3:] = np.array(
            [
                3.0 * mean_motion**2 * x + 2.0 * mean_motion * vy,
                -2.0 * mean_motion * vx,
                -mean_motion**2 * z,
            ]
        ) + external_acceleration
    return derivative

