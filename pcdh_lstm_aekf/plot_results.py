from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_results(sequence, fixed, adaptive, known, history, metrics, output_dir: Path) -> None:
    """Write all reproducibility curves to the requested fig directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    time = sequence["time"]
    colors = {"Fixed EKF": "#68737d", "Known-scale EKF": "#2f7d63", "PC-DH-LSTM-AEKF": "#c84b31"}
    results = {"Fixed EKF": fixed, "Known-scale EKF": known, "PC-DH-LSTM-AEKF": adaptive}

    plt.figure(figsize=(8.5, 4.5))
    for name, result in results.items():
        position_error = np.linalg.norm(sequence["truth"][:, :3] - result["estimate"][:, :3], axis=1)
        plt.plot(time, position_error, label=name, color=colors[name], linewidth=1.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Position error norm (m)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "01_position_error.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.5, 4.5))
    for name, result in results.items():
        velocity_error = np.linalg.norm(sequence["truth"][:, 3:] - result["estimate"][:, 3:], axis=1)
        plt.plot(time, velocity_error, label=name, color=colors[name], linewidth=1.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity error norm (m/s)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "02_velocity_error.png", dpi=180)
    plt.close()

    figure, axes = plt.subplots(2, 1, figsize=(8.5, 6.2), sharex=True)
    for name, result in results.items():
        normalized_nis = result["nis"] / np.maximum(result["dof"], 1)
        axes[0].plot(time, normalized_nis, label=name, color=colors[name], linewidth=0.9, alpha=0.9)
        axes[1].plot(time, result["nees"] / 6.0, label=name, color=colors[name], linewidth=0.9, alpha=0.9)
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("NIS / DOF")
    axes[1].set_ylabel("NEES / 6")
    axes[1].set_xlabel("Time (s)")
    axes[0].set_ylim(0.0, min(8.0, np.nanpercentile(adaptive["nis"] / np.maximum(adaptive["dof"], 1), 98) + 1.0))
    axes[1].set_yscale("log")
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    axes[0].legend(ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "03_nis_nees.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(8.5, 6.2), sharex=True)
    axis_names = ("x", "y", "z")
    measurement_names = ("rho", "theta", "psi")
    for axis, axis_name in enumerate(axis_names):
        measurement_name = measurement_names[axis]
        axes[0].plot(time, np.log10(sequence["q_psd"][:, axis] / sequence["q0_psd"][axis]), linewidth=1.0, label=f"true q-{axis_name}")
        axes[0].plot(time, np.log10(adaptive["q_estimate"][:, axis] / sequence["q0_psd"][axis]), linestyle="--", linewidth=1.0, label=f"estimated q-{axis_name}")
        axes[1].plot(time, np.log10(sequence["r_var"][:, axis] / sequence["r0_var"][axis]), linewidth=1.0, label=f"true R-{measurement_name}")
        axes[1].plot(time, np.log10(adaptive["r_estimate"][:, axis] / sequence["r0_var"][axis]), linestyle="--", linewidth=1.0, label=f"estimated R-{measurement_name}")
    axes[0].set_ylabel("log10 Q scale")
    axes[1].set_ylabel("log10 R scale")
    axes[1].set_xlabel("Time (s)")
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    axes[0].legend(ncol=3, fontsize=7)
    axes[1].legend(ncol=3, fontsize=7)
    figure.tight_layout()
    figure.savefig(output_dir / "04_noise_scales.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(3, 1, figsize=(8.5, 7.0), sharex=True)
    for axis, axis_name in enumerate(axis_names):
        axes[axis].plot(time, sequence["acceleration"][:, axis], label="true", color="#2f7d63")
        axes[axis].plot(time, adaptive["acceleration_estimate"][:, axis], label="LSTM", color="#c84b31", alpha=0.85)
        axes[axis].set_ylabel(f"a-{axis_name} (m/s2)")
        axes[axis].grid(alpha=0.25)
    axes[0].legend()
    axes[-1].set_xlabel("Time (s)")
    figure.tight_layout()
    figure.savefig(output_dir / "05_acceleration_head.png", dpi=180)
    plt.close(figure)

    names = list(metrics.keys())
    position_values = [metrics[name]["position_rss_rmse_m"] for name in names]
    velocity_values = [metrics[name]["velocity_rss_rmse_mps"] for name in names]
    figure, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    axes[0].bar(names, position_values, color=[colors[name] for name in names])
    axes[1].bar(names, velocity_values, color=[colors[name] for name in names])
    axes[0].set_ylabel("Position RSS RMSE (m)")
    axes[1].set_ylabel("Velocity RSS RMSE (m/s)")
    axes[0].tick_params(axis="x", rotation=18)
    axes[1].tick_params(axis="x", rotation=18)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "06_rmse_comparison.png", dpi=180)
    plt.close(figure)

    plt.figure(figsize=(7.0, 4.0))
    if isinstance(history, dict):
        training_history = history["training_loss"]
        validation_history = history["validation_loss"]
    else:
        training_history = history
        validation_history = None
    plt.plot(
        np.arange(1, len(training_history) + 1),
        training_history,
        color="#8b3f8f",
        marker="o",
        markersize=3,
        label="training",
    )
    if validation_history is not None:
        plt.plot(
            np.arange(1, len(validation_history) + 1),
            validation_history,
            color="#2f7d63",
            marker="s",
            markersize=3,
            label="validation",
        )
        plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "07_training_loss.png", dpi=180)
    plt.close()
