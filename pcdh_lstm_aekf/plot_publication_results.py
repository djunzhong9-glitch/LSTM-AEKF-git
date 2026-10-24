"""Chapter-results figures from archived, reproducible experiment outputs.

The stage plots join summaries of three disjoint intervals for visual comparison.
They must not be interpreted as sample-by-sample Monte Carlo trajectories.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


STYLE = {
    "Fixed EKF": ("#858C91", "-"),
    "GM-9D EKF": ("#2166AC", "-"),
    "Adaptive GM-AEKF": ("#8064A2", "-"),
    "PC-DH-LSTM-AEKF": ("#C94A4A", "-"),
    "GM-9D EKF + LSTM residual": ("#238B78", "-"),
    "GM-9D EKF + oracle residual": ("#B76518", "-"),
}
MAIN_METHODS = tuple(STYLE)[:5]
LEARNED = {"PC-DH-LSTM-AEKF", "GM-9D EKF + LSTM residual"}
STAGES = ("early", "middle", "late")
METRICS = (
    ("position_rss_rmse_m", "Position RSS RMSE (m)"),
    ("velocity_rss_rmse_mps", "Velocity RSS RMSE (m/s)"),
    ("normalized_nis", "NIS/DOF"),
    ("normalized_nees", "NEES/6"),
)


def configure() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def save(fig: plt.Figure, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    skill_scripts = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
    if skill_scripts.exists():
        sys.path.insert(0, str(skill_scripts))
        from audit_panel_alignment import require_matplotlib_panel_alignment

        require_matplotlib_panel_alignment(
            fig,
            json_out=out / f"{stem}.alignment.json",
            tolerance_pt=1.5,
            strict=True,
        )
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(out / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def interval(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or len(values) < 2:
        raise ValueError("Confidence interval requires at least two finite inference units")
    center = float(values.mean())
    half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return center, center - half, center + half


def inference_units(rows: pd.DataFrame, method: str, phase: str, metric: str) -> np.ndarray:
    frame = rows.loc[(rows.method == method) & (rows.phase == phase)]
    if method in LEARNED:
        values = frame.groupby("training_seed", sort=True)[metric].mean().to_numpy()
        if len(values) != 5:
            raise ValueError(f"Expected five training instances for {method}/{phase}")
    else:
        values = frame.sort_values("test_seed")[metric].to_numpy()
        if len(values) != 30:
            raise ValueError(f"Expected thirty test seeds for {method}/{phase}")
    return values


def quantile_curve(rows: pd.DataFrame, method: str, metric: str) -> np.ndarray:
    frame = rows.loc[(rows.method == method) & (rows.phase == "post_burn_in")]
    if method in LEARNED:
        frame = frame.groupby("test_seed", as_index=False)[metric].mean()
    values = np.sort(frame[metric].to_numpy(dtype=float))
    if len(values) != 30:
        raise ValueError(f"Expected thirty matched test trajectories for {method}")
    return values


def draw_training_and_tuning(root: Path, out: Path) -> None:
    ensemble = json.loads((root / "training_ensemble.json").read_text(encoding="utf-8"))
    tuning = json.loads((root / "gm_validation_tuning.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.95), layout="constrained")

    training = np.array([m["history"]["training_loss"] for m in ensemble["models"]])
    validation = np.array([m["history"]["validation_loss"] for m in ensemble["models"]])
    epochs = np.arange(1, training.shape[1] + 1)
    for curve in training:
        ax[0].plot(epochs, curve, color="#9BB3B0", lw=0.5, alpha=0.6)
    for curve in validation:
        ax[0].plot(epochs, curve, color="#D9A09E", lw=0.5, alpha=0.6)
    ax[0].plot(epochs, training.mean(axis=0), color="#238B78", lw=1.55, label="Training mean")
    ax[0].plot(epochs, validation.mean(axis=0), color="#C94A4A", lw=1.55, label="Validation mean")
    ax[0].set(xlabel="Epoch", ylabel="Sequence loss", title="a  Five independently trained networks")
    ax[0].legend(loc="upper right", ncol=1)
    ax[0].grid(alpha=0.18)

    grid = pd.DataFrame(tuning["candidate_grid"])
    for drive, part in grid.groupby("drive_psd_m2_s5", sort=True):
        part = part.sort_values("tau_s")
        ax[1].plot(part.tau_s, part.mean_relative_accuracy_score, marker="o", ms=2.6,
                   lw=1.1, label=f"{drive:.1g}")
    chosen = grid.loc[(grid.tau_s == tuning["selected_tau_s"]) &
                      np.isclose(grid.drive_psd_m2_s5, tuning["selected_drive_psd_m2_s5"], rtol=1e-8)]
    if len(chosen) != 1:
        raise ValueError("Frozen GM setting is absent from the validation grid")
    ax[1].scatter(chosen.tau_s, chosen.mean_relative_accuracy_score, s=65,
                  marker="*", color="#111111", zorder=5, label="Frozen choice")
    ax[1].set(xscale="log", xlabel="GM time constant τ (s)",
              ylabel="Validation accuracy score (lower is better)",
              title="b  Validation-only GM parameter search")
    ax[1].legend(title="Drive PSD (m²/s⁵)", loc="upper right", ncol=2,
                 fontsize=6.1, title_fontsize=6.1)
    ax[1].grid(alpha=0.18)
    save(fig, out, "16_training_and_gm_selection")


def draw_accuracy_distributions(rows: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), layout="constrained")
    for ax, (metric, label), letter in zip(axes, METRICS[:2], "ab"):
        for method in MAIN_METHODS:
            values = quantile_curve(rows, method, metric)
            color, linestyle = STYLE[method]
            x = np.r_[values[0], values]
            y = np.r_[0, np.arange(1, 31) / 30]
            ax.step(x, y, where="post", color=color, ls=linestyle, lw=1.5, label=method)
        ax.set(xlabel=label, ylabel="Empirical cumulative fraction",
               ylim=(0, 1.02), title=f"{letter}  Distribution over 30 matched test seeds")
        ax.grid(alpha=0.18)
    axes[0].legend(loc="lower right", fontsize=6.5)
    save(fig, out, "17_id_accuracy_distributions")


def paired_gain(rows: pd.DataFrame, method: str, metric: str) -> np.ndarray:
    gm = rows.loc[(rows.method == "GM-9D EKF") & (rows.phase == "post_burn_in"),
                  ["test_seed", metric]].set_index("test_seed")[metric]
    target = rows.loc[(rows.method == method) & (rows.phase == "post_burn_in")]
    if method in LEARNED:
        target = target.groupby("test_seed", as_index=False)[metric].mean()
    joined = gm.to_frame("gm").join(target.set_index("test_seed")[[metric]], how="inner")
    if len(joined) != 30:
        raise ValueError("Oracle and learned comparisons require the same thirty test seeds")
    return np.sort((joined.gm - joined[metric]).to_numpy())


def draw_oracle_gain(rows: pd.DataFrame, oracle_csv: Path, out: Path) -> None:
    oracle = pd.read_csv(oracle_csv)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
    for ax, (metric, label), letter in zip(axes, METRICS[:2], "ab"):
        gm = oracle.loc[oracle.method == "GM-9D EKF", ["test_seed", metric]].set_index("test_seed")
        ideal = oracle.loc[oracle.method == "GM-9D EKF + oracle residual",
                           ["test_seed", metric]].set_index("test_seed")
        gain_oracle = (gm[metric] - ideal[metric]).sort_values().to_numpy()
        gain_learned = paired_gain(rows, "GM-9D EKF + LSTM residual", metric)
        if len(gain_oracle) != 30:
            raise ValueError("Oracle diagnostic requires thirty paired test seeds")
        percentile = (np.arange(30) + 0.5) / 30 * 100
        ax.axhline(0, color="#555555", lw=0.8, ls="--")
        ax.plot(percentile, gain_oracle, color=STYLE["GM-9D EKF + oracle residual"][0],
                lw=1.65, marker="o", ms=2.2, label="Oracle residual")
        ax.plot(percentile, gain_learned, color=STYLE["GM-9D EKF + LSTM residual"][0],
                lw=1.65, marker="o", ms=2.2, label="LSTM residual")
        ax.set(xlabel="Test-seed percentile", ylabel=f"GM minus variant: {label}",
               title=f"{letter}  Paired gain over GM-9D")
        ax.grid(alpha=0.18)
    axes[0].legend(loc="upper left")
    save(fig, out, "18_oracle_residual_gain")


def stage_stats(rows: pd.DataFrame, method: str, metric: str) -> np.ndarray:
    return np.array([interval(inference_units(rows, method, phase, metric)) for phase in STAGES])


def draw_stage_consistency(rows: pd.DataFrame, out: Path) -> None:
    metrics = (
        ("normalized_nis", "NIS/DOF", 1.0),
        ("normalized_nees", "NEES/6", 1.0),
        ("nis_95_coverage", "NIS 95% coverage", 0.95),
        ("nees_95_coverage", "NEES 95% coverage", 0.95),
    )
    methods = MAIN_METHODS[1:]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), layout="constrained")
    x = np.arange(3)
    for ax, (metric, label, nominal), letter in zip(axes.flat, metrics, "abcd"):
        ax.axhline(nominal, color="#555555", lw=0.8, ls="--")
        for method in methods:
            stats = stage_stats(rows, method, metric)
            color, _ = STYLE[method]
            ax.plot(x, stats[:, 0], color=color, marker="o", ms=3, lw=1.35, label=method)
            ax.fill_between(x, stats[:, 1], stats[:, 2], color=color, alpha=0.11, lw=0)
        ax.set(xticks=x, xticklabels=["Early", "Middle", "Late"],
               ylabel=label, title=f"{letter}  {label}")
        ax.grid(alpha=0.17)
    axes[0, 0].legend(loc="upper left", fontsize=6.7)
    save(fig, out, "19_stage_consistency_bands")


def stage_paired_gain(rows: pd.DataFrame, metric: str, phase: str) -> np.ndarray:
    gm = rows.loc[(rows.method == "GM-9D EKF") & (rows.phase == phase),
                  ["test_seed", metric]].set_index("test_seed")
    residual = rows.loc[(rows.method == "GM-9D EKF + LSTM residual") & (rows.phase == phase),
                        ["test_seed", "training_seed", metric]]
    paired = residual.join(gm, on="test_seed", rsuffix="_gm")
    if len(paired) != 150:
        raise ValueError("Expected 5 training seeds x 30 matched test seeds")
    paired["difference"] = paired[metric] - paired[f"{metric}_gm"]
    return paired.groupby("training_seed", sort=True)["difference"].mean().to_numpy()


def draw_stage_accuracy(rows: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), layout="constrained")
    x = np.arange(3)
    for ax, (metric, label), letter in zip(axes.flat, METRICS, "abcd"):
        ax.axhline(0, color="#555555", lw=0.8, ls="--")
        stats = np.array([interval(stage_paired_gain(rows, metric, phase)) for phase in STAGES])
        ax.plot(x, stats[:, 0], color="#238B78", lw=1.5, marker="o", ms=3.2)
        ax.fill_between(x, stats[:, 1], stats[:, 2], color="#238B78", alpha=0.2)
        ax.set(xticks=x, xticklabels=["Early", "Middle", "Late"],
               ylabel=f"Residual minus GM: {label}", title=f"{letter}  Matched stage difference")
        ax.grid(alpha=0.18)
    save(fig, out, "20_stage_paired_differences")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render English manuscript results figures")
    parser.add_argument("--archive", type=Path, default=Path("artifacts/major_revision_v9"))
    parser.add_argument("--oracle", type=Path,
                        default=Path("artifacts/oracle_residual_diagnostics_v9/oracle_residual_metrics.csv"))
    parser.add_argument("--output", type=Path, default=Path("fig"))
    args = parser.parse_args()
    configure()
    rows = pd.read_csv(args.archive / "crossed_id_rows.csv")
    draw_training_and_tuning(args.archive, args.output)
    draw_accuracy_distributions(rows, args.output)
    draw_oracle_gain(rows, args.oracle, args.output)
    draw_stage_consistency(rows, args.output)
    draw_stage_accuracy(rows, args.output)


if __name__ == "__main__":
    main()
