"""Pair residual-fusion bar comparisons with oracle/learned gain curves."""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from redraw_additional_three import interval  # noqa: E402
from combine_ablation_oracle_gain import gain_curves  # noqa: E402

sys.path.insert(0, str(Path.home() / ".codex/skills/nature-figure/scripts"))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

mpl.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "svg.fonttype": "none",
                     "font.size": 15, "axes.labelsize": 15,
                     "xtick.labelsize": 15, "ytick.labelsize": 15,
                     "axes.linewidth": 0.85})


def draw():
    rows = pd.read_csv(ROOT / "artifacts/major_revision_v9/crossed_id_rows.csv")
    oracle = pd.read_csv(ROOT / "artifacts/oracle_residual_diagnostics_v9/oracle_residual_metrics.csv")
    post = rows[rows.phase == "post_burn_in"]
    fusion = "GM-9D EKF + LSTM residual"
    refs = ("GM-9D EKF", "GM-9D EKF + LSTM residual (no residual head)")
    labels = ("GM-9D\nreference", "Matched fusion\nwithout residual head")
    metrics = (("position_rss_rmse_m", "Position RMSE", "mm", "m"),
               ("velocity_rss_rmse_mps", "Velocity RMSE", "mm/s", "m/s"))
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.6))
    fig.subplots_adjust(left=0.13, right=0.985, top=0.89, bottom=0.16,
                        hspace=0.75, wspace=0.32)

    for j, (metric, ylabel, bar_unit, _) in enumerate(metrics):
        ax = axes[0, j]
        f = post.loc[post.method == fusion, ["training_seed", "test_seed", metric]]
        results = []
        for ref in refs:
            r = post.loc[post.method == ref, ["training_seed", "test_seed", metric]]
            keys = ["test_seed"] if ref == "GM-9D EKF" else ["training_seed", "test_seed"]
            joined = f.merge(r, on=keys, suffixes=("_f", "_r"))
            if len(joined) != 150:
                raise ValueError("Expected 5 training seeds × 30 shared test seeds")
            joined["difference"] = 1000 * (joined[f"{metric}_f"] - joined[f"{metric}_r"])
            train_key = "training_seed_f" if ref == "GM-9D EKF" else "training_seed"
            results.append((interval(joined.groupby(train_key).difference.mean()),
                            interval(joined.groupby("test_seed").difference.mean())))
        bounds = []
        for kind, color in enumerate(("#CB7127", "#4F82C4")):
            x = np.arange(2) + (-0.19 if kind == 0 else 0.19)
            v = [results[k][kind] for k in range(2)]
            mean = np.array([a[0] for a in v])
            lower = np.array([a[1] for a in v])
            upper = np.array([a[2] for a in v])
            bounds.extend(lower.tolist() + upper.tolist())
            ax.bar(x, mean, width=0.34, color=color, edgecolor="#555555",
                   linewidth=0.6, zorder=2)
            ax.errorbar(x, mean, yerr=np.vstack((mean - lower, upper - mean)),
                        fmt="none", ecolor="#252525", elinewidth=1.05,
                        capsize=3.0, capthick=1.05, zorder=4)
        span = max(bounds) - min(bounds)
        ax.set_ylim(min(0, min(bounds)) - .12 * span, max(0, max(bounds)) + .14 * span)
        ax.axhline(0, color="#333333", lw=1.0, zorder=3)
        ax.set_xticks(np.arange(2), labels, fontsize=15)
        ax.set_ylabel(f"{ylabel} difference ({bar_unit})")
        ax.set_title(f"{'ab'[j]}  Paired mean difference", loc="left", fontsize=15, pad=7)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis="y", color="#D7D7D7", linestyle="--", lw=.7, zorder=0)
        ax.tick_params(direction="in", top=True, right=True)

        ax = axes[1, j]
        ideal, learned = gain_curves(rows, oracle, metric)
        percentile = (np.arange(30) + 0.5) / 30 * 100
        ax.axhline(0, color="#555555", linestyle="--", lw=.9, zorder=1)
        ax.plot(percentile, ideal, color="#B76518", marker="o", ms=3,
                lw=1.7, label="Oracle residual", zorder=3)
        ax.plot(percentile, learned, color="#238B78", marker="o", ms=3,
                lw=1.7, label="LSTM residual", zorder=3)
        ax.set_xlim(-2, 102)
        ax.set_xlabel("Test-seed percentile")
        ax.set_ylabel(f"{ylabel} gain ({metrics[j][3]})")
        ax.set_title(f"{'cd'[j]}  Paired gain over GM-9D", loc="left", fontsize=15, pad=7)
        ax.grid(color="#D7D7D7", alpha=.65, lw=.7)
        ax.tick_params(direction="in", top=True, right=True)

    axes[1, 0].legend(loc="upper left", frameon=False, fontsize=15)
    fig.suptitle("Residual-fusion paired differences and attainable residual gains",
                 y=.978, fontsize=15)
    fig.legend(handles=[Patch(facecolor="#CB7127", edgecolor="#555555",
                              label="5 training-seed means · 30 fixed test trajectories"),
                        Patch(facecolor="#4F82C4", edgecolor="#555555",
                              label="30 test-seed means · 5 fixed trained models")],
               loc="center", bbox_to_anchor=(.5, .518), ncol=1, frameon=False, fontsize=15)
    fig.text(.5, .055, "Top: descriptive 95% CIs; negative differences favor residual fusion.\nBottom: 30 matched test seeds.",
             ha="center", fontsize=15)
    fig.canvas.draw()
    stem = "residual_fusion_and_oracle_gain_combined_15pt"
    require_matplotlib_panel_alignment(fig, json_out=HERE / f"{stem}.alignment.json")
    fig.savefig(HERE / f"{stem}.png", dpi=600, facecolor="white")
    fig.savefig(HERE / f"{stem}.pdf", facecolor="white")
    fig.savefig(HERE / f"{stem}.svg", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw()
