"""Redraw four existing result figures as reference-style bars with 95% CIs."""

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SKILL_SCRIPTS = Path.home() / ".codex/skills/nature-figure/scripts"
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402


METRICS = (
    ("position_rss_rmse_m", "Position RSS RMSE (m)", "a  Position accuracy"),
    ("velocity_rss_rmse_mps", "Velocity RSS RMSE (m/s)", "b  Velocity accuracy"),
    ("normalized_nis", "NIS / DOF", "c  Measurement consistency"),
    ("normalized_nees", "NEES / 6", "d  State consistency"),
)
PALETTE = ("#A7A7A7", "#D1A83D", "#65A34C", "#4F82C4", "#E25B4A", "#EF9145", "#8866AC")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0.85,
    "axes.edgecolor": "#454545",
    "axes.labelcolor": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def stats(items):
    mean = np.asarray([x["mean"] for x in items], dtype=float)
    lower = np.asarray([x["ci_lower"] for x in items], dtype=float)
    upper = np.asarray([x["ci_upper"] for x in items], dtype=float)
    if not (np.isfinite(mean).all() and np.isfinite(lower).all() and np.isfinite(upper).all()):
        raise ValueError("Nonfinite plotted value")
    if not ((lower <= mean).all() and (mean <= upper).all()):
        raise ValueError("Invalid confidence interval")
    return mean, lower, upper


def panel(ax, items, labels, colors, metric_label, panel_title, *, difference=False,
          target=None, rotated=False, note=None):
    x = np.arange(len(items))
    mean, lower, upper = stats(items)
    ax.bar(x, mean, width=0.72, color=colors, edgecolor=["#555555"] * len(items),
           linewidth=0.55, zorder=2)
    ax.errorbar(x, mean, yerr=np.vstack((mean - lower, upper - mean)), fmt="none",
                ecolor="#252525", elinewidth=1.05, capsize=3.0, capthick=1.05, zorder=4)
    if difference:
        ax.axhline(0, color="#333333", lw=1.0, zorder=3)
    if target is not None:
        ax.axhline(target, color="#777777", linestyle=(0, (4, 3)), lw=0.9, zorder=1)
    lo = min(float(lower.min()), 0.0)
    hi = max(float(upper.max()), 0.0)
    span = hi - lo
    if span < 1e-12:
        span = max(abs(hi), 1.0) * 0.1
    ax.set_ylim(lo - 0.14 * span if difference else 0, hi + 0.20 * span)
    ax.set_xlim(-0.65, len(items) - 0.35)
    ax.set_xticks(x, labels, rotation=50 if rotated else 0,
                  rotation_mode="anchor", ha="right" if rotated else "center",
                  fontsize=7.5 if rotated else 8.3)
    ax.set_ylabel(metric_label, fontsize=9.5)
    ax.set_title(panel_title, loc="left", fontsize=10.5, pad=8)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
    ax.grid(axis="y", color="#D7D7D7", linestyle="--", linewidth=0.72, zorder=0)
    ax.tick_params(direction="in", top=True, right=True, length=3.5)
    if note:
        ax.text(0.98, 0.96, note, transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color="#404040")


def save(fig, stem, subtitle, title_fontsize=10.5):
    fig.suptitle(subtitle, x=0.5, y=0.985, fontsize=title_fontsize)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=OUT / f"{stem}.alignment.json")
    fig.savefig(OUT / f"{stem}.png", dpi=600, facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", facecolor="white")
    plt.close(fig)


def canvas(figsize=(12.6, 8.6)):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.89, bottom=0.15,
                        wspace=0.22, hspace=0.43)
    return fig, axes


def draw_08():
    data = read("artifacts/experiment_matrix.json")
    paired = data["paired_difference_vs_full"]
    names = ("Shared NIS Q/R in full model", "No acceleration head",
             "No R fast channel", "No Q slow constraint")
    labels = ("Shared NIS\nQ/R", "No accel.\nhead", "No fast R\nchannel", "No slow Q\nconstraint")
    colors = (PALETTE[3], PALETTE[4], PALETTE[1], PALETTE[2])
    fig, axes = canvas()
    for ax, (key, ylabel, title) in zip(axes.flat, METRICS):
        panel(ax, [paired[n][key] for n in names], labels, colors, ylabel, title,
              difference=True)
    fig.legend([Patch(facecolor=c, edgecolor="#555555") for c in colors], names,
               loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=2,
               frameon=False, fontsize=8.4)
    save(fig, "08_ablation_paired_differences",
         "Ablation − full model · 30 paired test seeds · bars: mean; whiskers: 95% t CI")


def draw_09():
    data = read("artifacts/experiment_matrix.json")
    summary = data["summary"]
    names = tuple(summary)
    labels = ("Fixed", "Innov.-CM", "Sage–Husa", "Coupled", "GM-9D", "GM+R-CM",
              "Known scale", "PC-DH", "GM+Res", "Res no head", "Res no R",
              "Shared Q/R", "No accel.", "No fast R", "No slow Q", "Det. accel.",
              "Feature switch")
    color_map = {"PC-DH-LSTM-AEKF": PALETTE[4], "No R fast channel": PALETTE[1],
                 "GM-9D EKF": PALETTE[3], "GM-9D EKF + R-CM": PALETTE[6],
                 "Known-scale EKF": PALETTE[2]}
    colors = [color_map.get(n, PALETTE[0]) for n in names]
    specs = (("normalized_nis", "NIS / DOF", "a  Measurement consistency", 1.0),
             ("normalized_nees", "NEES / 6", "b  State consistency", 1.0),
             ("nis_95_coverage", "NIS 95% coverage", "c  Measurement coverage", 0.95),
             ("nees_95_coverage", "NEES 95% coverage", "d  State coverage", 0.95))
    fig, axes = canvas((18.0, 9.4))
    fig.subplots_adjust(bottom=0.19, hspace=0.62, wspace=0.17)
    for ax, (key, ylabel, title, target) in zip(axes.flat, specs):
        panel(ax, [summary[n][key] for n in names], labels, colors, ylabel, title,
              target=target, rotated=True)
    fig.legend(handles=[Patch(facecolor=c, edgecolor="#555555", label=l) for c, l in (
        (PALETTE[0], "Other methods"), (PALETTE[3], "GM-9D"),
        (PALETTE[6], "GM-9D + R-CM"), (PALETTE[2], "Known scale"),
        (PALETTE[4], "PC-DH"), (PALETTE[1], "No fast R"))],
        loc="lower center", bbox_to_anchor=(0.5, 0.008), ncol=6, frameon=False, fontsize=8.5)
    save(fig, "09_consistency_comparison",
         "Monte Carlo consistency · 30 test seeds · bars: mean; whiskers: 95% t CI · dashed: nominal")


def draw_10():
    data = read("artifacts/generalization_gap.json")
    paired = data["paired_gap"]
    names = tuple(paired)
    abbreviations = ("Fixed", "Innov.-CM", "Sage–Husa", "Coupled\nNIS Q/R", "GM-9D",
                     "GM+R-CM", "Known\nscale", "PC-DH", "Shared\nNIS Q/R", "No accel.",
                     "No fast R", "No slow Q", "No shadow\nFixed")
    color_map = {"PC-DH-LSTM-AEKF": PALETTE[4], "No R fast channel": PALETTE[1],
                 "GM-9D EKF": PALETTE[3], "GM-9D EKF + R-CM": PALETTE[6],
                 "Known-scale EKF": PALETTE[2]}
    colors = [color_map.get(n, PALETTE[0]) for n in names]
    fig, axes = canvas((16.5, 9.0))
    fig.subplots_adjust(bottom=0.18, hspace=0.58, wspace=0.18)
    for ax, (key, ylabel, title) in zip(axes.flat, METRICS):
        panel(ax, [paired[n][key] for n in names], abbreviations, colors, ylabel, title,
              difference=True, rotated=True)
    handles = [Patch(facecolor=c, edgecolor="#555555", label=l) for c, l in (
        (PALETTE[0], "Other methods"), (PALETTE[3], "GM-9D"),
        (PALETTE[6], "GM-9D + R-CM"), (PALETTE[2], "Known scale"),
        (PALETTE[4], "PC-DH"), (PALETTE[1], "No fast R"))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.006),
               ncol=6, frameon=False, fontsize=8.5)
    save(fig, "10_id_ood_generalization_gap",
         "OOD − ID · 30 paired test seeds · bars: mean; whiskers: 95% t CI")


def draw_12():
    data = read("artifacts/major_revision_v9/crossed_id.json")
    names = ("Fixed EKF", "GM-9D EKF", "Adaptive GM-AEKF", "PC-DH-LSTM-AEKF",
             "GM-9D EKF + LSTM residual")
    labels = ("Fixed", "GM-9D", "Adaptive\nGM", "PC-DH", "GM +\nresidual")
    colors = (PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[4], PALETTE[3])
    fig, axes = canvas()
    for ax, (key, ylabel, title) in zip(axes.flat, METRICS):
        items = []
        for name in names:
            values = data["summary"][name]
            if "across_training_seeds" in values:
                values = values["across_training_seeds"]
            items.append(values["post_burn_in"][key])
        panel(ax, items, labels, colors, ylabel, title,
              target=1.0 if key.startswith("normalized_") else None)
    fig.legend([Patch(facecolor=c, edgecolor="#555555") for c in colors], names,
               loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=3,
               frameon=False, fontsize=8.3)
    save(fig, "12_training_seed_summary",
         "In-distribution · 5 training seeds · bars: mean; whiskers: 95% CI")


def draw_13():
    data = read("artifacts/major_revision_v9/crossed_id.json")
    tests = data["confirmatory_ablation_tests"]["tests"]
    names = tuple(dict.fromkeys(t["method"] for t in tests))
    labels = ("GM+Res", "No residual\nhead", "With LSTM\nR head", "Shared\nNIS Q/R",
              "No accel.\nhead", "No fast R\nchannel", "No slow Q\nconstraint", "Det. accel.\nmean only")
    colors = (PALETTE[3], PALETTE[2], PALETTE[6], PALETTE[0],
              PALETTE[4], PALETTE[1], PALETTE[2], PALETTE[0])
    fig, axes = canvas((14.7, 8.8))
    for ax, (key, ylabel, title) in zip(axes.flat, METRICS):
        selected = {t["method"]: t for t in tests if t["metric"] == key}
        items = [selected[n]["difference_summary"] for n in names]
        panel(ax, items, labels, colors, ylabel, title, difference=True)
        for i, name in enumerate(names):
            if selected[name]["reject_at_0_05"]:
                ax.annotate("*", (i, items[i]["ci_upper"]), xytext=(0, 4),
                            textcoords="offset points", ha="center", fontsize=9,
                            color="#5E3D91")
    fig.legend(handles=[Patch(facecolor=c, edgecolor="#555555", label=l) for c, l in (
        (PALETTE[3], "GM + residual"), (PALETTE[2], "Component change"),
        (PALETTE[6], "With LSTM R head"), (PALETTE[0], "Other ablations"),
        (PALETTE[4], "No acceleration head"), (PALETTE[1], "No fast R"))],
        loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False, fontsize=8.5)
    save(fig, "13_holm_ablation_training_seeds",
         "Ablation − full model · 5 training seeds · 95% CI · * Holm-adjusted p < 0.05")


def draw_14():
    mpl.rcParams.update({"font.size": 15, "axes.labelsize": 15,
                         "xtick.labelsize": 15, "ytick.labelsize": 15})
    data = read("artifacts/major_revision_v9/crossed_id.json")
    names = ("Fixed EKF", "Innovation-CM AEKF", "Coupled-NIS Q/R AEKF", "GM-9D EKF",
             "GM-9D EKF + R-CM", "Adaptive GM-AEKF", "PC-DH-LSTM-AEKF",
             "GM-9D EKF + LSTM residual", "No R fast channel")
    labels = ("Fixed", "R-CM", "Coupled", "GM-9D", "GM+R", "AGM", "PC-DH",
              "GM+Res", "No R-fast")
    colors = (PALETTE[0], PALETTE[0], PALETTE[0], PALETTE[1], PALETTE[6],
              PALETTE[2], PALETTE[4], PALETTE[3], PALETTE[5])
    specs = (("normalized_nis", "NIS / DOF", "a  Measurement consistency", 1.0),
             ("normalized_nees", "NEES / 6", "b  State consistency", 1.0),
             ("nis_95_coverage", "NIS 95% coverage", "c  Measurement coverage", 0.95),
             ("nees_95_coverage", "NEES 95% coverage", "d  State coverage", 0.95))
    fig, axes = canvas((15.0, 11.0))
    fig.subplots_adjust(left=0.10, right=0.985, top=0.89, bottom=0.26,
                        wspace=0.27, hspace=0.95)
    for ax, (key, ylabel, title, target) in zip(axes.flat, specs):
        items = []
        for name in names:
            values = data["summary"][name]
            if "across_training_seeds" in values:
                values = values["across_training_seeds"]
            items.append(values["post_burn_in"][key])
        panel(ax, items, labels, colors, ylabel, title, target=target)
        ax.tick_params(axis="both", labelsize=15)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.xaxis.label.set_fontsize(15)
        ax.yaxis.label.set_fontsize(15)
        ax.set_title(title, loc="left", fontsize=15, pad=8)
    fig.legend(handles=[Patch(facecolor=c, edgecolor="#555555", label=l) for c, l in (
        (PALETTE[0], "Classical"), (PALETTE[1], "GM-9D"), (PALETTE[6], "GM + R-CM"),
        (PALETTE[2], "Adaptive GM"), (PALETTE[4], "PC-DH"),
        (PALETTE[3], "GM + residual"), (PALETTE[5], "No fast R"))],
        loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=3, frameon=False, fontsize=15)
    save(fig, "14_consistency_coverage_15pt",
         "In-distribution consistency and coverage · post-burn-in · 95% CI · dashed: nominal",
         title_fontsize=15)


def draw_15():
    data = read("artifacts/major_revision_v9/crossed_id.json")
    tests = data["adaptive_gm_ablation_tests"]["tests"]
    names = tuple(dict.fromkeys(t["method"] for t in tests))
    labels = tuple(n.removeprefix("Adaptive GM-AEKF ").strip("()") for n in names)
    labels = tuple(s.replace(" ", "\n", 1) for s in labels)
    colors = (PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[4], PALETTE[3])
    fig, axes = canvas()
    for ax, (key, ylabel, title) in zip(axes.flat, METRICS):
        selected = {t["method"]: t for t in tests if t["metric"] == key}
        items = [selected[n]["difference_summary"] for n in names]
        panel(ax, items, labels, colors, ylabel, title, difference=True)
        for i, name in enumerate(names):
            if selected[name]["reject_at_0_05"]:
                value = items[i]["ci_upper"]
                ax.annotate("*", (i, value), xytext=(0, 5), textcoords="offset points",
                            ha="center", fontsize=10, color="#5E3D91")
    fig.legend([Patch(facecolor=c, edgecolor="#555555") for c in colors],
               [n.removeprefix("Adaptive GM-AEKF ").strip("()") for n in names],
               loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=3,
               frameon=False, fontsize=8.3)
    save(fig, "15_adaptive_gm_ablation",
         "Variant − Adaptive GM · 30 paired test seeds · 95% t CI · * Holm-adjusted p < 0.05")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    draw_08()
    draw_09()
    draw_10()
    draw_12()
    draw_13()
    draw_14()
    draw_15()
