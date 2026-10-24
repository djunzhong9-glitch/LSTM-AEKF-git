"""Four-panel GM diagnostic from the archived, unsmoothed test trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .plot_publication_results import configure, save


def draw(trace: dict[str, np.ndarray], output: Path) -> None:
    time = np.asarray(trace["time"], dtype=float)
    truth = trace["truth"]
    fig, grid = plt.subplots(2, 2, figsize=(7.2, 4.8), sharex=True)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.94, bottom=0.115,
                        wspace=0.25, hspace=0.39)
    a, b, c, d = grid.flat

    feature_delta = trace["main_feature_norm"] - trace["shadow_feature_norm"]
    a.plot(time, feature_delta, color="#C94A4A", lw=0.55)
    a.axhline(0, color="#555555", lw=0.7)
    a.set(ylabel="Feature-norm difference", title="a  Feature-source switch",
          ylim=(-0.72, 0.72))

    for key, label, color in (
        ("acceleration", "Truth", "#333333"),
        ("gm_acceleration", "GM state", "#2166AC"),
        ("residual_acceleration", "GM + LSTM", "#238B78"),
    ):
        b.plot(time, 1e3 * trace[key][:, 0], color=color, lw=0.7, label=label)
    b.axhline(1.35, color="#999999", ls=":", lw=0.65)
    b.axhline(-1.35, color="#999999", ls=":", lw=0.65)
    b.set(ylabel="Acceleration x (mm/s²)",
          title="b  Physical acceleration", ylim=(-1.55, 1.82))
    b.legend(loc="upper right", ncol=2, fontsize=6.5, handlelength=1.4,
             columnspacing=0.7, labelspacing=0.2, frameon=True,
             facecolor="white", edgecolor="none", framealpha=0.88)

    errors: dict[str, np.ndarray] = {}
    for key, label, color in (
        ("gm_estimate", "GM-9D EKF", "#2166AC"),
        ("residual_estimate", "GM + LSTM", "#238B78"),
        ("adaptive_gm_estimate", "Adaptive GM", "#8064A2"),
    ):
        errors[key] = np.linalg.norm(truth[:, :3] - trace[key][:, :3], axis=1)
        c.plot(time, errors[key], color=color, lw=0.7, label=label)
    c.set(xlabel="Time (s)", ylabel="Position error (m)",
          title="c  GM-family position errors", ylim=(0, 0.39))
    c.legend(loc="upper right", ncol=1, fontsize=6.5, handlelength=1.4,
             labelspacing=0.2, frameon=True, facecolor="white",
             edgecolor="none", framealpha=0.88)

    for key, label, color in (
        ("residual_estimate", "GM + LSTM", "#238B78"),
        ("adaptive_gm_estimate", "Adaptive GM", "#8064A2"),
    ):
        delta_cm = 100 * (errors[key] - errors["gm_estimate"])
        d.plot(time, delta_cm, color=color, lw=0.65, label=label)
    d.axhline(0, color="#555555", lw=0.7)
    d.set(xlabel="Time (s)", ylabel="Difference vs GM-9D (cm)",
          title="d  Paired position-error differences", ylim=(-7, 7))
    d.legend(loc="upper right", ncol=2, fontsize=6.5, handlelength=1.4,
             columnspacing=0.7, labelspacing=0.2, frameon=True,
             facecolor="white", edgecolor="none", framealpha=0.88)

    for axis in grid.flat:
        axis.set_xlim(time[0], time[-1])
        axis.set_xticks(np.arange(0, 251, 50))
        axis.axvline(25, color="#777777", ls="--", lw=0.65)
        axis.grid(axis="x", alpha=0.12)
    save(fig, output, "26_gm_mechanism_2x2")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot four raw GM trajectory panels")
    parser.add_argument("--archive", type=Path, default=Path("artifacts/major_revision_v9"))
    parser.add_argument("--seed", type=int, default=22260719)
    parser.add_argument("--training-seed", type=int, default=20260719)
    parser.add_argument("--output", type=Path, default=Path("fig"))
    args = parser.parse_args()

    configure()
    source = args.archive / (
        f"method_figure_trace_seed_{args.seed}_train_{args.training_seed}.npz"
    )
    with np.load(source) as archive:
        trace = {key: archive[key] for key in archive.files}
    draw(trace, args.output)


if __name__ == "__main__":
    main()
