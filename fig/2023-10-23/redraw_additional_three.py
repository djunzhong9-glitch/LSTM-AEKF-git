"""Reference-style versions of the paired residual and confirmatory ablation plots."""

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy.stats import t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from redraw_reference_style import PALETTE, read  # noqa: E402

sys.path.insert(0, str(Path.home() / ".codex/skills/nature-figure/scripts"))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

mpl.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "svg.fonttype": "none",
                     "font.size": 15, "axes.labelsize": 15,
                     "xtick.labelsize": 15, "ytick.labelsize": 15,
                     "axes.linewidth": 0.85})


def save(fig, stem):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=HERE / f"{stem}.alignment.json")
    fig.savefig(HERE / f"{stem}.png", dpi=600, facecolor="white")
    fig.savefig(HERE / f"{stem}.pdf", facecolor="white")
    fig.savefig(HERE / f"{stem}.svg", facecolor="white")
    plt.close(fig)


def interval(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    half = float(t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n))
    return mean, mean - half, mean + half


def axes_style(ax, ylabel, title, labels, lo, hi):
    span = hi - lo
    ax.set_ylim(min(0, lo) - 0.14 * span, max(0, hi) + 0.16 * span)
    ax.axhline(0, color="#333333", lw=1.0, zorder=3)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=15, pad=8)
    ax.set_xticks(np.arange(len(labels)), labels, fontsize=15)
    if len(labels) > 2:
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis="y", color="#D7D7D7", linestyle="--", lw=0.75, zorder=0)
    ax.tick_params(direction="in", top=True, right=True, length=3.5)


def draw_residual_comparison():
    rows = pd.read_csv(ROOT / "artifacts/major_revision_v9/crossed_id_rows.csv")
    rows = rows[rows.phase == "post_burn_in"]
    fusion = "GM-9D EKF + LSTM residual"
    references = ("GM-9D EKF", "GM-9D EKF + LSTM residual (no residual head)")
    labels = ("GM-9D\nreference", "Matched fusion\nwithout residual head")
    metrics = (("position_rss_rmse_m", "Position RMSE difference (mm)", "a  Position accuracy"),
               ("velocity_rss_rmse_mps", "Velocity RMSE difference (mm/s)", "b  Velocity accuracy"))
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))
    fig.subplots_adjust(left=0.12, right=0.985, top=0.81, bottom=0.30, wspace=0.30)
    for ax, (metric, ylabel, title) in zip(axes, metrics):
        f = rows.loc[rows.method == fusion, ["training_seed", "test_seed", metric]]
        results = []
        for ref in references:
            r = rows.loc[rows.method == ref, ["training_seed", "test_seed", metric]]
            keys = ["test_seed"] if ref == "GM-9D EKF" else ["training_seed", "test_seed"]
            joined = f.merge(r, on=keys, suffixes=("_f", "_r"))
            if len(joined) != 150:
                raise ValueError("Expected 5 training seeds × 30 matched test seeds")
            joined["difference"] = 1000 * (joined[f"{metric}_f"] - joined[f"{metric}_r"])
            train_key = "training_seed_f" if ref == "GM-9D EKF" else "training_seed"
            results.append((interval(joined.groupby(train_key).difference.mean()),
                            interval(joined.groupby("test_seed").difference.mean())))
        for j, color in enumerate(("#CB7127", PALETTE[3])):
            centers = np.arange(2) + (-0.19 if j == 0 else 0.19)
            values = [results[i][j] for i in range(2)]
            mean = np.array([v[0] for v in values])
            lower = np.array([v[1] for v in values])
            upper = np.array([v[2] for v in values])
            ax.bar(centers, mean, width=0.34, color=color, edgecolor="#555555",
                   linewidth=0.6, zorder=2)
            ax.errorbar(centers, mean, yerr=np.vstack((mean - lower, upper - mean)),
                        fmt="none", ecolor="#252525", capsize=3.2,
                        elinewidth=1.1, capthick=1.1, zorder=4)
        all_bounds = [v for pair in results for item in pair for v in item[1:]]
        axes_style(ax, ylabel, title, labels, min(all_bounds), max(all_bounds))
    fig.suptitle("Residual fusion − reference · descriptive paired differences and 95% CIs",
                 fontsize=15, y=0.98)
    fig.legend(handles=[Patch(facecolor="#CB7127", edgecolor="#555555",
                              label="5 training-seed means · 30 fixed test trajectories"),
                        Patch(facecolor=PALETTE[3], edgecolor="#555555",
                              label="30 test-seed means · 5 fixed trained models")],
               loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=1,
               frameon=False, fontsize=15)
    save(fig, "residual_fusion_paired_comparisons_bars_15pt")


def draw_confirmatory_ablation():
    data = read("artifacts/major_revision_v9/crossed_id.json")
    tests = data["confirmatory_ablation_tests"]["tests"]
    names = ("Shared NIS Q/R in full model", "No acceleration head",
             "No R fast channel", "No Q slow constraint",
             "Deterministic acceleration mean only")
    labels = ("Shared NIS\nQ/R", "No accel.\nhead", "No fast R\nchannel",
              "No slow Q\nconstraint", "Det. accel.\nonly")
    colors = (PALETTE[3], PALETTE[4], PALETTE[1], PALETTE[2], PALETTE[5])
    metrics = (("position_rss_rmse_m", "Position RMSE difference (mm)", "a  Position accuracy", 1000),
               ("velocity_rss_rmse_mps", "Velocity RMSE difference (mm/s)", "b  Velocity accuracy", 1000),
               ("normalized_nis", "NIS / DOF difference", "c  Measurement consistency", 1),
               ("normalized_nees", "NEES / 6 difference", "d  State consistency", 1))
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    fig.subplots_adjust(left=0.12, right=0.985, top=0.89, bottom=0.29,
                        wspace=0.28, hspace=0.95)
    for ax, (metric, ylabel, title, scale) in zip(axes.flat, metrics):
        selected = {x["method"]: x for x in tests if x["metric"] == metric}
        data_values = [selected[n]["difference_summary"] for n in names]
        mean = scale * np.array([v["mean"] for v in data_values])
        lower = scale * np.array([v["ci_lower"] for v in data_values])
        upper = scale * np.array([v["ci_upper"] for v in data_values])
        ax.bar(np.arange(5), mean, width=0.72, color=colors, edgecolor="#555555",
               linewidth=0.6, zorder=2)
        ax.errorbar(np.arange(5), mean,
                    yerr=np.vstack((mean - lower, upper - mean)), fmt="none",
                    ecolor="#252525", elinewidth=1.1, capsize=3.0,
                    capthick=1.1, zorder=4)
        axes_style(ax, ylabel, title, labels, float(lower.min()), float(upper.max()))
    assert all(selected["reject_at_0_05"] for selected in tests if selected["method"] in names)
    fig.suptitle("Ablation − PC-DH-LSTM-AEKF · 5 training seeds × 30 fixed test seeds",
                 fontsize=15, y=0.985)
    fig.legend([Patch(facecolor=c, edgecolor="#555555") for c in colors], names,
               loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=2,
               frameon=False, fontsize=15)
    fig.text(0.5, 0.17, "Bars: paired mean; whiskers: 95% CI. All 20 Holm-adjusted p values < 0.05.",
             ha="center", fontsize=15)
    save(fig, "confirmatory_ablation_five_variants_bars_15pt")
    rows = [{"method": x["method"], "metric": x["metric"],
             "holm_adjusted_p_value": x["holm_adjusted_p_value"]}
            for x in tests if x["method"] in names]
    pd.DataFrame(rows).to_csv(HERE / "confirmatory_ablation_five_variants_pvalues.csv", index=False)


if __name__ == "__main__":
    draw_residual_comparison()
    draw_confirmatory_ablation()
