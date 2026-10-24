from pathlib import Path

import numpy as np


def save_arrays(sequence, fixed, adaptive, known, output_path: Path) -> None:
    """Persist truth, estimates and learned parameters for later analysis."""

    np.savez_compressed(
        output_path,
        time=sequence["time"],
        truth=sequence["truth"],
        measurements=sequence["measurements"],
        measurement_mask=sequence["mask"],
        true_acceleration=sequence["acceleration"],
        true_q_psd=sequence["q_psd"],
        true_r_var=sequence["r_var"],
        fixed_estimate=fixed["estimate"],
        adaptive_estimate=adaptive["estimate"],
        known_scale_estimate=known["estimate"],
        adaptive_acceleration=adaptive["acceleration_estimate"],
        adaptive_q_psd=adaptive["q_estimate"],
        adaptive_r_var=adaptive["r_estimate"],
    )

