"""Time-resolved mechanism illustrations for one archived test trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .make_config import make_config
from .plot_publication_results import configure, save
from .run_adaptive_ekf import precompute_lstm_outputs, run_adaptive_ekf
from .run_adaptive_gm_ekf import apply_adaptive_gm_parameters, run_adaptive_gm_ekf
from .run_fixed_ekf import run_fixed_ekf
from .run_gm_ekf import run_gm_9d_ekf
from .run_gm_residual_lstm_ekf import run_gm_residual_lstm_ekf
from .save_model import load_model
from .simulate_sequence import simulate_sequence


def smooth(values: np.ndarray, steps: int = 21) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rolling(
        steps, center=True, min_periods=1
    ).mean().to_numpy()


def colored_line(axis, time, data, label, color, *, smooth_steps=21):
    axis.plot(time, smooth(data, smooth_steps), color=color, lw=1.15, label=label)


def draw_pcdh(time, sequence, fixed, pcdh, config, output_dir):
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 6.35), sharex=True)
    fig.subplots_adjust(left=0.12, right=0.72, top=0.95, bottom=0.08, hspace=0.34)
    error_fixed = np.linalg.norm(sequence["truth"][:, :3] - fixed["estimate"][:, :3], axis=1)
    error_pcdh = np.linalg.norm(sequence["truth"][:, :3] - pcdh["estimate"][:, :3], axis=1)
    colored_line(axes[0], time, error_fixed, "Fixed EKF", "#858C91")
    colored_line(axes[0], time, error_pcdh, "PC-DH-LSTM-AEKF", "#C94A4A")
    axes[0].set(ylabel="Position error (m)", title="a  Six-dimensional learning compensation")

    colored_line(axes[1], time, sequence["acceleration"][:, 0] * 1e3,
                 "True", "#333333")
    colored_line(axes[1], time, pcdh["acceleration_estimate"][:, 0] * 1e3,
                 "LSTM head", "#C94A4A")
    axes[1].set(ylabel="Acceleration x (mm/s²)", title="b  Acceleration head")

    colored_line(axes[2], time, sequence["r_var"][:, 0] / config.r0_var[0],
                 "True / nominal", "#333333")
    colored_line(axes[2], time, pcdh["r_estimate"][:, 0] / config.r0_var[0],
                 "Estimated / nominal", "#C94A4A")
    axes[2].set(ylabel="Range R scale", title="c  Fast measurement-noise channel")

    colored_line(axes[3], time, sequence["q_psd"][:, 0] / config.q0_psd[0],
                 "True / nominal", "#333333")
    colored_line(axes[3], time, pcdh["q_estimate"][:, 0] / config.q0_psd[0],
                 "Estimated / nominal", "#C94A4A")
    axes[3].set(xlabel="Time (s)", ylabel="Process Q scale",
                title="d  Slow structured process-noise channel")

    for axis in axes:
        axis.grid(alpha=0.16)
        axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
        axis.axvline(25, color="#777777", ls="--", lw=0.6)
    save(fig, output_dir, "21_pcdh_mechanism_trajectory")


def draw_gm(time, sequence, fixed, no_shadow, gm, residual, adaptive_gm,
            config, output_dir):
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 6.35), sharex=True)
    fig.subplots_adjust(left=0.12, right=0.72, top=0.95, bottom=0.08, hspace=0.34)
    shadow_norm = np.linalg.norm(fixed["features"][:, :3], axis=1)
    no_shadow_norm = np.linalg.norm(no_shadow["features"][:, :3], axis=1)
    colored_line(axes[0], time, shadow_norm, "Shadow feature source", "#2166AC")
    colored_line(axes[0], time, no_shadow_norm, "Main-filter feature source", "#C94A4A")
    axes[0].set(ylabel="Feature norm", title="a  Feature-source switch, fixed weights")

    colored_line(axes[1], time, sequence["acceleration"][:, 0] * 1e3,
                 "True", "#333333")
    colored_line(axes[1], time, gm["acceleration_estimate"][:, 0] * 1e3,
                 "GM state", "#2166AC")
    colored_line(axes[1], time, residual["acceleration_estimate"][:, 0] * 1e3,
                 "GM + LSTM residual", "#238B78")
    axes[1].set(ylabel="Acceleration x (mm/s²)",
                title="b  Physical state and learned residual")

    for result, label, color in (
        (gm, "GM-9D EKF", "#2166AC"),
        (residual, "GM + LSTM residual", "#238B78"),
        (adaptive_gm, "Adaptive GM-AEKF", "#8064A2"),
    ):
        error = np.linalg.norm(sequence["truth"][:, :3] - result["estimate"][:, :3], axis=1)
        colored_line(axes[2], time, error, label, color)
    axes[2].set(ylabel="Position error (m)", title="c  GM-family estimation errors")

    drive = np.asarray(adaptive_gm["drive_psd_estimate"], dtype=float)
    drive = drive.mean(axis=1) if drive.ndim == 2 else drive
    colored_line(axes[3], time, adaptive_gm["tau_estimate"] / config.gm_tau_s,
                 "Time constant / frozen τ", "#8064A2")
    colored_line(axes[3], time, drive / config.gm_drive_psd,
                 "Drive PSD / frozen q", "#B76518")
    axes[3].axhline(1, color="#555555", ls=":", lw=0.75)
    axes[3].set(xlabel="Time (s)", ylabel="Relative parameter",
                title="d  Adaptive GM parameter response")

    for axis in axes:
        axis.grid(alpha=0.16)
        axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
        axis.axvline(25, color="#777777", ls="--", lw=0.6)
    save(fig, output_dir, "22_gm_mechanism_trajectory")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render single-seed method diagnostics")
    parser.add_argument("--archive", type=Path, default=Path("artifacts/major_revision_v9"))
    parser.add_argument("--seed", type=int, default=22260719)
    parser.add_argument("--training-seed", type=int, default=20260719)
    parser.add_argument("--output", type=Path, default=Path("fig"))
    args = parser.parse_args()

    configure()
    config = make_config("paper")
    gm_tuning = json.loads((args.archive / "gm_validation_tuning.json").read_text(encoding="utf-8"))
    config.gm_tau_s = float(gm_tuning["selected_tau_s"])
    config.gm_drive_psd = float(gm_tuning["selected_drive_psd_m2_s5"])
    adaptive_tuning = json.loads(
        (args.archive / "adaptive_gm_validation_tuning.json").read_text(encoding="utf-8")
    )
    apply_adaptive_gm_parameters(config, adaptive_tuning["selected_parameters"])

    model = load_model(args.archive / "models" /
                       f"dual_head_lstm_trainseed_{args.training_seed}.npz")
    residual_model = load_model(args.archive / "models" /
                                f"gm_residual_lstm_trainseed_{args.training_seed}.npz")
    sequence = simulate_sequence(config, args.seed, "nonlinear_structured")
    fixed = run_fixed_ekf(sequence, config)
    model_outputs = precompute_lstm_outputs(model, fixed["features"])
    pcdh = run_adaptive_ekf(sequence, model, config,
                            precomputed_shadow_features=fixed["features"],
                            precomputed_model_outputs=model_outputs)
    no_shadow = run_adaptive_ekf(sequence, model, config,
                                 use_shadow_fixed_features=False)
    gm = run_gm_9d_ekf(sequence, config)
    residual = run_gm_residual_lstm_ekf(sequence, residual_model, config,
                                         use_lstm_r_head=False,
                                         use_nis_feedback=False,
                                         use_r_cm_correction=True)
    adaptive_gm = run_adaptive_gm_ekf(sequence, config)

    draw_pcdh(sequence["time"], sequence, fixed, pcdh, config, args.output)
    draw_gm(sequence["time"], sequence, fixed, no_shadow, gm, residual,
            adaptive_gm, config, args.output)
    np.savez_compressed(
        args.archive / f"method_figure_trace_seed_{args.seed}_train_{args.training_seed}.npz",
        time=sequence["time"], truth=sequence["truth"], acceleration=sequence["acceleration"],
        true_q_psd=sequence["q_psd"], true_r_var=sequence["r_var"],
        fixed_estimate=fixed["estimate"], pcdh_estimate=pcdh["estimate"],
        gm_estimate=gm["estimate"], residual_estimate=residual["estimate"],
        adaptive_gm_estimate=adaptive_gm["estimate"],
        pcdh_acceleration=pcdh["acceleration_estimate"],
        gm_acceleration=gm["acceleration_estimate"],
        residual_acceleration=residual["acceleration_estimate"],
        pcdh_q_psd=pcdh["q_estimate"], pcdh_r_var=pcdh["r_estimate"],
        adaptive_gm_tau=adaptive_gm["tau_estimate"],
        adaptive_gm_drive=adaptive_gm["drive_psd_estimate"],
        shadow_feature_norm=np.linalg.norm(fixed["features"][:, :3], axis=1),
        main_feature_norm=np.linalg.norm(no_shadow["features"][:, :3], axis=1),
    )


if __name__ == "__main__":
    main()
