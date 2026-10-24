"""Combine the ablation bars and paired oracle-gain curves from their source data."""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from redraw_reference_style import METRICS, PALETTE, panel, read  # noqa: E402

sys.path.insert(0, str(Path.home() / ".codex/skills/nature-figure/scripts"))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

mpl.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "svg.fonttype": "none"})


def gain_curves(rows, oracle, metric):
    gm = oracle.loc[oracle.method == "GM-9D EKF", ["test_seed", metric]].set_index("test_seed")
    ideal = oracle.loc[oracle.method == "GM-9D EKF + oracle residual",
                       ["test_seed", metric]].set_index("test_seed")
    gain_oracle = (gm[metric] - ideal[metric]).sort_values().to_numpy()

    gm_rows = rows.loc[(rows.method == "GM-9D EKF") & (rows.phase == "post_burn_in"),
                       ["test_seed", metric]].set_index("test_seed")[metric]
    learned = rows.loc[(rows.method == "GM-9D EKF + LSTM residual") &
                       (rows.phase == "post_burn_in")]
    learned = learned.groupby("test_seed", as_index=False)[metric].mean()
    paired = gm_rows.to_frame("gm").join(learned.set_index("test_seed")[[metric]], how="inner")
    gain_learned = np.sort((paired.gm - paired[metric]).to_numpy())
    if len(gain_oracle) != 30 or len(gain_learned) != 30:
        raise ValueError("Expected 30 matched test seeds for both gain curves")
    return gain_oracle, gain_learned


def main():
    matrix = read("artifacts/experiment_matrix.json")
    paired = matrix["paired_difference_vs_full"]
    names = ("Shared NIS Q/R in full model", "No acceleration head",
             "No R fast channel", "No Q slow constraint")
    labels = ("Shared NIS\nQ/R", "No accel.\nhead", "No fast R\nchannel",
              "No slow Q\nconstraint")
    colors = (PALETTE[3], PALETTE[4], PALETTE[1], PALETTE[2])

    rows = pd.read_csv(ROOT / "artifacts/major_revision_v9/crossed_id_rows.csv")
    oracle = pd.read_csv(ROOT / "artifacts/oracle_residual_diagnostics_v9/oracle_residual_metrics.csv")

    fig, axes = plt.subplots(3, 2, figsize=(12.6, 12.2))
    fig.subplots_adjust(left=0.095, right=0.98, top=0.895, bottom=0.105,
                        hspace=0.52, wspace=0.24)
    for ax, (key, ylabel, title) in zip(axes[:2].flat, METRICS):
        panel(ax, [paired[n][key] for n in names], labels, colors, ylabel, title,
              difference=True)

    percentile = (np.arange(30) + 0.5) / 30 * 100
    for index, (ax, (metric, ylabel, _)) in enumerate(zip(axes[2], METRICS[:2])):
        oracle_curve, learned_curve = gain_curves(rows, oracle, metric)
        ax.axhline(0, color="#555555", linestyle="--", lw=1.0, zorder=1)
        ax.plot(percentile, oracle_curve, color="#B76518", lw=1.7, marker="o",
                ms=3.3, label="Oracle residual", zorder=3)
        ax.plot(percentile, learned_curve, color="#238B78", lw=1.7, marker="o",
                ms=3.3, label="LSTM residual", zorder=3)
        ax.set(xlim=(-2, 102), xlabel="Test-seed percentile",
               ylabel=f"GM-9D minus variant: {ylabel}",
               title=f"{'ef'[index]}  Paired gain over GM-9D")
        ax.title.set_ha("left")
        ax.title.set_position((0, 1.02))
        ax.grid(color="#D7D7D7", alpha=0.65, linewidth=0.7)
        ax.tick_params(direction="in", top=True, right=True, length=3.5)
    axes[2, 0].legend(loc="upper left", frameon=False, fontsize=8.5)

    fig.suptitle("Ablation effects and residual-correction gains across matched test seeds",
                 fontsize=13, y=0.978)
    fig.legend([Patch(facecolor=c, edgecolor="#555555") for c in colors], names,
               loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=2,
               frameon=False, fontsize=8.5)
    fig.canvas.draw()
    stem = "08_ablation_and_18_oracle_gain_combined"
    require_matplotlib_panel_alignment(fig, json_out=HERE / f"{stem}.alignment.json")
    fig.savefig(HERE / f"{stem}.png", dpi=600, facecolor="white")
    fig.savefig(HERE / f"{stem}.pdf", facecolor="white")
    fig.savefig(HERE / f"{stem}.svg", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
