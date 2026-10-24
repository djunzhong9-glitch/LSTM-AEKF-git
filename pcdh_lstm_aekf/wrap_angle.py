import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap an azimuth residual to (-pi, pi]."""

    #把角度归一化(-pi,pi]
    wrapped = (angle + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.pi if wrapped <= -np.pi else wrapped)
