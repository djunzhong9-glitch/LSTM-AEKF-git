"""Compact, unsmoothed GM diagnostic from the archived test trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from .plot_publication_results import configure, save


def _candidate_strip(values: np.ndarray, low: str, high: str) -> tuple[np.ndarray, float, float, int]:
    levels, counts = np.unique(np.round(values, 8), return_counts=True)
    major = np.sort(levels[np.argsort(counts)[-2:]])
    if len(major) != 2:
        raise ValueError("Expected two dominant adaptive-GM candidates")
    low_value, high_value = map(float, major)
    colors = np.tile(mcolors.to_rgb("#888888"), (len(values), 1))
    low_mask = np.isclose(values, low_value, rtol=0, atol=1e-7)
    high_mask = np.isclose(values, high_value, rtol=0, atol=1e-7)
    colors[low_mask] = mcolors.to_rgb(low)
    colors[high_mask] = mcolors.to_rgb(high)
    return colors, low_value, high_value, int((~(low_mask | high_mask)).sum())


def draw(trace: dict[str, np.ndarray], tau_s: float, drive_psd: float, output: Path) -> None:
    time = np.asarray(trace["time"], dtype=float)
    truth = trace["truth"]
    fig, axes = plt.subplots(
        5, 1, figsize=(7.2, 7.15), sharex=True,
        gridspec_kw={"height_ratios": [0.9, 1.3, 1.0, 0.66, 0.38]},
    )
    fig.subplots_adjust(left=0.14, right=0.98, top=0.965, bottom=0.09, hspace=0.62)

    feature_delta = trace["main_feature_norm"] - trace["shadow_feature_norm"]
    axes[0].plot(time, feature_delta, color="#C94A4A", lw=0.55)
    axes[0].axhline(0, color="#555555", lw=0.7)
    axes[0].set(
        ylabel="Main - shadow\nfeature norm",
        title="a  Feature-source difference (raw samples)",
    )
    axes[0].set_ylim(-0.72, 0.72)

    for key, label, color in (
        ("acceleration", "Truth", "#333333"),
        ("gm_acceleration", "GM state", "#2166AC"),
        ("residual_acceleration", "GM + LSTM", "#238B78"),
    ):
        axes[1].plot(time, 1e3 * trace[key][:, 0], color=color, lw=0.7, label=label)
    axes[1].axhline(1.35, color="#999999", ls=":", lw=0.65, label="Clip limit")
    axes[1].axhline(-1.35, color="#999999", ls=":", lw=0.65)
    axes[1].set(
        ylabel="Acceleration x (mm/s²)",
        title="b  Physical acceleration and estimates",
        ylim=(-1.55, 1.82),
    )
    axes[1].legend(
        loc="upper center", ncol=4, fontsize=6.7, handlelength=1.5,
        columnspacing=0.8, labelspacing=0.2,
        frameon=True, facecolor="white", edgecolor="none", framealpha=0.88,
    )

    errors: dict[str, np.ndarray] = {}
    for key, label, color in (
        ("gm_estimate", "GM-9D EKF", "#2166AC"),
        ("residual_estimate", "GM + LSTM", "#238B78"),
        ("adaptive_gm_estimate", "Adaptive GM", "#8064A2"),
    ):
        error = np.linalg.norm(truth[:, :3] - trace[key][:, :3], axis=1)
        errors[key] = error
        axes[2].plot(time, error, color=color, lw=0.7, label=label)
    axes[2].set(ylabel="Position error (m)", title="c  GM-family position errors", ylim=(0, 0.39))
    axes[2].legend(
        loc="upper right", ncol=3, fontsize=6.7, handlelength=1.5,
        columnspacing=0.8, labelspacing=0.2,
        frameon=True, facecolor="white", edgecolor="none", framealpha=0.88,
    )

    for key, color in (
        ("residual_estimate", "#238B78"),
        ("adaptive_gm_estimate", "#8064A2"),
    ):
        paired_delta_cm = 100 * (errors[key] - errors["gm_estimate"])
        axes[3].plot(time, paired_delta_cm, color=color, lw=0.6)
    axes[3].axhline(0, color="#555555", lw=0.7)
    axes[3].set(ylabel="Δ vs GM (cm)", ylim=(-7, 7))

    drive = np.asarray(trace["adaptive_gm_drive"], dtype=float)
    drive = drive.mean(axis=1) if drive.ndim == 2 else drive
    tau_strip, tau_low, tau_high, tau_other = _candidate_strip(
        trace["adaptive_gm_tau"] / tau_s, "#E2D9EF", "#6B4090"
    )
    drive_strip, drive_low, drive_high, drive_other = _candidate_strip(
        drive / drive_psd, "#F7DDC4", "#A84B07"
    )
    image = np.stack((tau_strip, drive_strip), axis=0)
    dt = float(np.median(np.diff(time)))
    axes[4].imshow(
        image, aspect="auto", interpolation="nearest", origin="upper",
        extent=(time[0] - dt / 2, time[-1] + dt / 2, 0, 2),
    )
    axes[4].set(
        ylabel="Candidate",
        title="d  Adaptive GM selections (light = low, dark = high)",
        xlabel="Time (s)",
    )
    axes[4].set_yticks(
        (0.5, 1.5),
        (f"q ratio {drive_low:g}/{drive_high:g}",
         f"tau ratio {tau_low:g}/{tau_high:g}"),
    )
    axes[4].tick_params(axis="y", length=0, labelsize=6.7)

    for axis in axes:
        axis.set_xlim(time[0], time[-1])
        axis.set_xticks(np.arange(0, 251, 50))
        axis.axvline(25, color="#777777", ls="--", lw=0.65)
        axis.grid(axis="x", alpha=0.12)
    axes[4].grid(False)
    fig.text(
        0.98, 0.015,
        f"Discrete candidates: {tau_other} transitional τ samples; "
        f"{drive_other} transitional q samples; no smoothing.",
        ha="right", va="bottom", fontsize=5.5, color="#555555",
    )
    save(fig, output, "25_gm_mechanism_compact")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot compact raw GM mechanism diagnostic")
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
