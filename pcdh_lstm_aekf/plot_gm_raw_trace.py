"""Audit plot for the archived GM mechanism trajectory, without smoothing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .plot_publication_results import configure, save


def draw(trace: dict[str, np.ndarray], tau_s: float, drive_psd: float, output: Path) -> None:
    time = trace["time"]
    truth = trace["truth"]
    fig, axes = plt.subplots(
        5, 1, figsize=(7.2, 7.35), sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 0.52, 0.52]},
    )
    fig.subplots_adjust(left=0.13, right=0.72, top=0.965, bottom=0.07, hspace=0.42)

    axes[0].plot(time, trace["shadow_feature_norm"], color="#2166AC", lw=0.6,
                 label="Shadow feature source")
    axes[0].plot(time, trace["main_feature_norm"], color="#C94A4A", lw=0.6,
                 label="Main-filter feature source")
    axes[0].set(ylabel="Feature norm", title="a  Feature-source switch (raw samples)")

    for key, label, color in (
        ("acceleration", "True", "#333333"),
        ("gm_acceleration", "GM state", "#2166AC"),
        ("residual_acceleration", "GM + LSTM residual", "#238B78"),
    ):
        axes[1].plot(time, 1e3 * trace[key][:, 0], color=color, lw=0.7, label=label)
    axes[1].axhline(1.35, color="#999999", ls=":", lw=0.65)
    axes[1].axhline(-1.35, color="#999999", ls=":", lw=0.65,
                    label="Simulation clip limit")
    axes[1].set(ylabel="Acceleration x (mm/s²)",
                title="b  Physical acceleration and GM estimates")

    for key, label, color in (
        ("gm_estimate", "GM-9D EKF", "#2166AC"),
        ("residual_estimate", "GM + LSTM residual", "#238B78"),
        ("adaptive_gm_estimate", "Adaptive GM-AEKF", "#8064A2"),
    ):
        error = np.linalg.norm(truth[:, :3] - trace[key][:, :3], axis=1)
        axes[2].plot(time, error, color=color, lw=0.7, label=label)
    axes[2].set(ylabel="Position error (m)", title="c  GM-family estimation errors")

    drive = np.asarray(trace["adaptive_gm_drive"], dtype=float)
    drive = drive.mean(axis=1) if drive.ndim == 2 else drive
    axes[3].scatter(time, trace["adaptive_gm_tau"] / tau_s,
                    color="#8064A2", s=1.0, linewidths=0,
                    label="Selected time constant")
    axes[3].axhline(1, color="#555555", ls=":", lw=0.75)
    axes[3].set(ylabel="τ / τ frozen", title="d  Adaptive GM candidate selections",
                ylim=(0.5, 8.5), yticks=(1, 8))
    axes[4].scatter(time, drive / drive_psd,
                    color="#B76518", s=1.0, linewidths=0,
                    label="Selected drive PSD")
    axes[4].axhline(1, color="#555555", ls=":", lw=0.75)
    axes[4].set(xlabel="Time (s)", ylabel="q / q frozen",
                ylim=(0.7, 1.35), yticks=(0.8, 1.0, 1.25))

    for axis in axes:
        axis.set_xlim(time[0], time[-1])
        axis.set_xticks(np.arange(0, 251, 50))
        axis.axvline(25, color="#777777", ls="--", lw=0.6)
        axis.grid(alpha=0.16)
        axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    save(fig, output, "24_gm_mechanism_raw")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot raw GM trajectory from saved arrays")
    parser.add_argument(
        "--archive", type=Path, default=Path("artifacts/major_revision_v9")
    )
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
    tuning = json.loads(
        (args.archive / "gm_validation_tuning.json").read_text(encoding="utf-8")
    )
    draw(
        trace,
        float(tuning["selected_tau_s"]),
        float(tuning["selected_drive_psd_m2_s5"]),
        args.output,
    )


if __name__ == "__main__":
    main()
