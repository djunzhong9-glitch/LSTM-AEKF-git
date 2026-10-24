import numpy as np

from .true_dynamics import true_dynamics


def rk4_step(
    state: np.ndarray,
    dt: float,
    mean_motion: float,
    semimajor_axis: float,
    mu: float,
    external_acceleration: np.ndarray,
    nonlinear: bool,
) -> np.ndarray:
    """Advance relative dynamics by one fourth-order Runge-Kutta step."""

    arguments = (mean_motion, semimajor_axis, mu, external_acceleration, nonlinear)
    k1 = true_dynamics(state, *arguments)
    k2 = true_dynamics(state + 0.5 * dt * k1, *arguments)
    k3 = true_dynamics(state + 0.5 * dt * k2, *arguments)
    k4 = true_dynamics(state + dt * k3, *arguments)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

