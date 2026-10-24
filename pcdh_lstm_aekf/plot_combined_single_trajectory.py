"""Recompose the four archived single-trajectory diagnostics into one figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .make_config import make_config


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "artifacts" / "test_results.npz"
OUTPUT = ROOT / "fig" / "23_single_trajectory_combined"

FILTERS = (
    ("Fixed EKF", "fixed_estimate", "#68737d"),
    ("Known-scale EKF", "known_scale_estimate", "#2f7d63"),
    ("PC-DH-LSTM-AEKF", "adaptive_estimate", "#c84b31"),
)
AXIS_COLORS = ("#2475b7", "#248a4a", "#9561ba")


def main() -> None:
    with np.load(SOURCE) as archive:
        data = {key: archive[key] for key in archive.files}
    config = make_config("paper")
    time = data["time"]
    truth = data["truth"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure = plt.figure(figsize=(12.8, 9.2))
    grid = figure.add_gridspec(
        3,
        6,
        left=0.075,
        right=0.99,
        bottom=0.125,
        top=0.925,
        hspace=0.58,
        wspace=0.72,
        height_ratios=(1.0, 0.92, 1.1),
    )
    position = figure.add_subplot(grid[0, :3])
    velocity = figure.add_subplot(grid[0, 3:])
    acceleration = [figure.add_subplot(grid[1, 2 * axis : 2 * axis + 2]) for axis in range(3)]
    process = figure.add_subplot(grid[2, :3])
    measurement = figure.add_subplot(grid[2, 3:])

    for name, key, color in FILTERS:
        estimate = data[key]
        position.plot(time, np.linalg.norm(truth[:, :3] - estimate[:, :3], axis=1), color=color, lw=1.2, label=name)
        velocity.plot(time, np.linalg.norm(truth[:, 3:] - estimate[:, 3:], axis=1), color=color, lw=1.2)
    position.set_title("(a) Position error")
    velocity.set_title("(b) Velocity error")
    position.set_ylabel("Error norm (m)")
    velocity.set_ylabel("Error norm (m/s)")

    for axis, plot in enumerate(acceleration):
        name = "xyz"[axis]
        plot.plot(time, 1e3 * data["true_acceleration"][:, axis], color="#2f7d63", lw=1.5, label="True")
        plot.plot(time, 1e3 * data["adaptive_acceleration"][:, axis], color="#c84b31", lw=1.2, label="LSTM")
        plot.set_title(f"({chr(ord('c') + axis)}) Acceleration {name}")
        plot.set_ylabel(r"mm/s$^2$")
    acceleration[0].legend(loc="upper right", frameon=False, fontsize=11)

    for axis, color in enumerate(AXIS_COLORS):
        process.plot(
            time,
            np.log10(data["true_q_psd"][:, axis] / config.q0_psd[axis]),
            color=color,
            lw=1.3,
        )
        process.plot(
            time,
            np.log10(data["adaptive_q_psd"][:, axis] / config.q0_psd[axis]),
            color=color,
            ls="--",
            lw=1.1,
        )
        measurement.plot(
            time,
            np.log10(data["true_r_var"][:, axis] / config.r0_var[axis]),
            color=color,
            lw=1.3,
        )
        measurement.plot(
            time,
            np.log10(data["adaptive_r_var"][:, axis] / config.r0_var[axis]),
            color=color,
            ls="--",
            lw=1.1,
        )
    process.set_title("(f) Process-noise scale")
    measurement.set_title("(g) Measurement-noise scale")
    process.set_ylabel(r"$\log_{10}(Q/Q_0)$")
    measurement.set_ylabel(r"$\log_{10}(R/R_0)$")

    for plot in (position, velocity, *acceleration, process, measurement):
        plot.set_xlim(time[0], time[-1])
        plot.set_xticks(np.arange(0, 251, 50))
        plot.grid(alpha=0.23, lw=0.7)
    for plot in (process, measurement):
        plot.set_xlabel("Time (s)")

    filter_handles = [Line2D([0], [0], color=color, lw=2) for _, _, color in FILTERS]
    figure.legend(
        filter_handles,
        [name for name, _, _ in FILTERS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        fontsize=13,
    )
    scale_handles = [Line2D([0], [0], color=color, lw=2) for color in AXIS_COLORS]
    scale_handles.extend(
        [
            Line2D([0], [0], color="#555555", lw=2),
            Line2D([0], [0], color="#555555", lw=2, ls="--"),
        ]
    )
    figure.legend(
        scale_handles,
        ("x / range", "y / elevation", "z / azimuth", "True", "Estimated"),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=5,
        frameon=False,
        fontsize=12,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for extension, kwargs in (
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 300}),
        ("tiff", {"dpi": 300}),
    ):
        figure.savefig(OUTPUT.with_suffix(f".{extension}"), facecolor="white", **kwargs)
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
