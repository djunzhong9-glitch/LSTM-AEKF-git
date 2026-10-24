import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, ScalarFormatter
import numpy as np


ABLATION_METHODS = (
    "Shared NIS Q/R in full model",
    "No acceleration head",
    "No R fast channel",
    "No Q slow constraint",
    "No shadow Fixed EKF",
)

METHOD_LABELS = {
    "Fixed EKF": "Fixed EKF",
    "Innovation-CM AEKF": "Innovation-CM AEKF",
    "Sage-Husa/Fading AEKF": "Sage-Husa/Fading AEKF",
    "Coupled-NIS Q/R AEKF": "Coupled-NIS Q/R AEKF",
    "GM-9D EKF": "GM-9D EKF",
    "GM-9D EKF + R-CM": "GM-9D EKF + R-CM",
    "Adaptive GM-AEKF": "Adaptive GM-AEKF",
    "GM-9D EKF + LSTM residual": "GM-9D EKF + LSTM residual",
    "GM-9D EKF + LSTM residual (no residual head)": "GM-9D EKF + LSTM residual (no residual head)",
    "GM-9D EKF + LSTM residual (no LSTM R head)": "GM-9D EKF + LSTM residual (no LSTM R head)",
    "Known-scale EKF": "Known-scale EKF",
    "PC-DH-LSTM-AEKF": "Full PC-DH-LSTM-AEKF",
    "Shared NIS Q/R in full model": "Shared NIS Q/R",
    "No acceleration head": "No acceleration head",
    "No R fast channel": "No R fast channel",
    "No Q slow constraint": "No Q slow constraint",
    "No shadow Fixed EKF": "No shadow Fixed EKF",
}

ABLATION_COLORS = {
    "Shared NIS Q/R in full model": "#4C78A8",
    "No acceleration head": "#E45756",
    "No R fast channel": "#F2A541",
    "No Q slow constraint": "#59A14F",
    "No shadow Fixed EKF": "#B279A2",
}

PAIR_METRICS = (
    ("position_rss_rmse_m", "Position RSS RMSE difference (m)", "(a) Position accuracy"),
    ("velocity_rss_rmse_mps", "Velocity RSS RMSE difference (m/s)", "(b) Velocity accuracy"),
    ("normalized_nis", "NIS/DOF difference", "(c) Measurement consistency"),
    ("normalized_nees", "NEES/6 difference", "(d) State consistency"),
)

CONSISTENCY_METRICS = (
    ("normalized_nis", "NIS/DOF", 1.0, "(a) NIS/DOF"),
    ("normalized_nees", "NEES/6", 1.0, "(b) NEES/6"),
    ("nis_95_coverage", "NIS 95% coverage", 0.95, "(c) NIS coverage"),
    ("nees_95_coverage", "NEES 95% coverage", 0.95, "(d) NEES coverage"),
)


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#4A4A4A",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#222222",
            "text.color": "#202020",
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
        }
    )


def _save_figure(figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = output_dir / f"{stem}.{suffix}"
        save_kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix == "png":
            save_kwargs["dpi"] = 320
        figure.savefig(path, **save_kwargs)
        paths.append(path)
    return paths


def _format_delta_axis(axis, values: np.ndarray) -> None:
    maximum = float(np.max(np.abs(values))) if values.size else 1.0
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    if maximum < 0.01:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        axis.xaxis.set_major_formatter(formatter)
    else:
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def _padded_limits(lower: np.ndarray, upper: np.ndarray, reference: float) -> tuple[float, float]:
    minimum = min(float(np.min(lower)), reference)
    maximum = max(float(np.max(upper)), reference)
    span = maximum - minimum
    if span <= 1e-12:
        span = max(abs(reference), 1.0) * 0.1
    padding = 0.14 * span
    return minimum - padding, maximum + padding


def plot_paired_ablation_differences(matrix: dict, output_dir: Path) -> list[Path]:
    """Plot paired ablation-minus-full means with two-sided 95% t intervals."""

    paired = matrix["paired_difference_vs_full"]
    ablation_methods = tuple(method for method in ABLATION_METHODS if method in paired)
    if not ablation_methods:
        raise KeyError("No paired ablation results found in the input matrix")

    figure, axes = plt.subplots(2, 2, figsize=(12.4, 7.9), sharey=True)
    positions = np.arange(len(ablation_methods))

    for panel_index, (axis, (metric, xlabel, panel_title)) in enumerate(
        zip(axes.flat, PAIR_METRICS, strict=True)
    ):
        means = np.asarray([paired[method][metric]["mean"] for method in ablation_methods])
        lowers = np.asarray([paired[method][metric]["ci_lower"] for method in ablation_methods])
        uppers = np.asarray([paired[method][metric]["ci_upper"] for method in ablation_methods])
        significant = (lowers > 0.0) | (uppers < 0.0)

        axis.axvline(0.0, color="#303030", linestyle="--", linewidth=1.0, zorder=1)
        for index, method in enumerate(ablation_methods):
            color = ABLATION_COLORS[method]
            axis.errorbar(
                means[index],
                positions[index],
                xerr=np.array(
                    [[means[index] - lowers[index]], [uppers[index] - means[index]]]
                ),
                fmt="o",
                markersize=6.8,
                markerfacecolor=color if significant[index] else "white",
                markeredgecolor=color,
                markeredgewidth=1.5,
                ecolor=color,
                elinewidth=1.6,
                capsize=3.5,
                capthick=1.3,
                zorder=3,
            )

        axis.set_xlim(*_padded_limits(lowers, uppers, 0.0))
        _format_delta_axis(axis, np.concatenate((lowers, uppers)))
        axis.set_xlabel(xlabel, fontsize=10.5)
        axis.set_title(panel_title, loc="left", fontsize=11.5, fontweight="normal")
        axis.set_ylim(len(ablation_methods) - 0.5, -0.5)
        axis.set_yticks(positions)
        if panel_index % 2 == 0:
            axis.set_yticklabels(
                [METHOD_LABELS.get(method, method) for method in ablation_methods], fontsize=9.5
            )
        else:
            axis.tick_params(axis="y", labelleft=False)
        axis.grid(axis="x")
        axis.tick_params(axis="x", labelsize=9)
        axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        f"Paired same-seed ablation differences ({matrix['runs']} Monte Carlo runs)",
        fontsize=15,
        fontweight="normal",
        y=0.985,
    )
    figure.text(
        0.5,
        0.945,
        "Difference = ablation - full PC-DH-LSTM-AEKF; points are means and bars are two-sided 95% t intervals",
        ha="center",
        va="center",
        fontsize=9.8,
        color="#555555",
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="#4A4A4A",
            markerfacecolor="#4A4A4A",
            linestyle="none",
            markersize=6.5,
            label="95% CI excludes 0",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="#4A4A4A",
            markerfacecolor="white",
            linestyle="none",
            markersize=6.5,
            label="95% CI includes 0",
        ),
        Line2D(
            [0],
            [0],
            color="#303030",
            linestyle="--",
            linewidth=1.0,
            label="No difference from full model",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
        fontsize=9.2,
    )
    figure.subplots_adjust(left=0.25, right=0.975, top=0.88, bottom=0.12, hspace=0.42, wspace=0.20)
    paths = _save_figure(figure, output_dir, "08_ablation_paired_differences")
    plt.close(figure)
    return paths


def _method_style(method: str) -> tuple[str, str, float]:
    if method == "PC-DH-LSTM-AEKF":
        return "#C44E52", "D", 7.2
    if method == "GM-9D EKF":
        return "#4C78A8", "P", 6.8
    if method == "GM-9D EKF + R-CM":
        return "#7A5195", "X", 6.8
    if method == "No R fast channel":
        return "#E69F00", "s", 6.6
    if method == "Known-scale EKF":
        return "#2A9D8F", "^", 6.6
    return "#687780", "o", 5.5


def plot_consistency_comparison(matrix: dict, output_dir: Path) -> list[Path]:
    """Plot four Monte Carlo consistency metrics for every method."""

    summary = matrix["summary"]
    methods = tuple(summary.keys())
    positions = np.arange(len(methods))
    figure, axes = plt.subplots(2, 2, figsize=(13.4, 9.6), sharey=True)

    for panel_index, (axis, (metric, xlabel, target, panel_title)) in enumerate(
        zip(axes.flat, CONSISTENCY_METRICS, strict=True)
    ):
        means = np.asarray([summary[method][metric]["mean"] for method in methods])
        lowers = np.asarray([summary[method][metric]["ci_lower"] for method in methods])
        uppers = np.asarray([summary[method][metric]["ci_upper"] for method in methods])
        axis.axvline(target, color="#303030", linestyle="--", linewidth=1.0, zorder=1)

        for index, method in enumerate(methods):
            color, marker, marker_size = _method_style(method)
            axis.errorbar(
                means[index],
                positions[index],
                xerr=np.array(
                    [[means[index] - lowers[index]], [uppers[index] - means[index]]]
                ),
                fmt=marker,
                markersize=marker_size,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.7,
                ecolor=color,
                elinewidth=1.25,
                capsize=2.7,
                capthick=1.0,
                alpha=1.0 if method in {"PC-DH-LSTM-AEKF", "No R fast channel", "Known-scale EKF"} else 0.78,
                zorder=3 if method == "PC-DH-LSTM-AEKF" else 2,
            )

        axis.set_xlim(*_padded_limits(lowers, uppers, target))
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        if "coverage" in metric:
            axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        else:
            axis.xaxis.set_major_formatter(FormatStrFormatter("%.1f" if metric == "normalized_nees" else "%.2f"))
        axis.set_xlabel(xlabel, fontsize=10.5)
        axis.set_title(panel_title, loc="left", fontsize=11.5, fontweight="normal")
        axis.set_ylim(len(methods) - 0.5, -0.5)
        axis.set_yticks(positions)
        if panel_index % 2 == 0:
            axis.set_yticklabels([METHOD_LABELS.get(method, method) for method in methods], fontsize=9.2)
        else:
            axis.tick_params(axis="y", labelleft=False)
        axis.grid(axis="x")
        axis.tick_params(axis="x", labelsize=9)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            0.985,
            0.025,
            f"Reference {target:g}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            color="#666666",
        )

    figure.suptitle(
        f"Monte Carlo comparison of consistency metrics ({matrix['runs']} random seeds)",
        fontsize=15,
        fontweight="normal",
        y=0.985,
    )
    figure.text(
        0.5,
        0.949,
        "Points are cross-seed means; bars are two-sided 95% t intervals; dashed lines denote nominal references",
        ha="center",
        va="center",
        fontsize=9.8,
        color="#555555",
    )
    legend_handles = [
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#C44E52", markeredgecolor="white", markersize=7.2, label="Full model"),
        Line2D([0], [0], marker="P", color="none", markerfacecolor="#4C78A8", markeredgecolor="white", markersize=6.8, label="GM-9D"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#7A5195", markeredgecolor="white", markersize=6.8, label="GM-9D + R-CM"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#E69F00", markeredgecolor="white", markersize=6.8, label="No R fast channel"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#2A9D8F", markeredgecolor="white", markersize=6.8, label="Known-scale reference"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#687780", markeredgecolor="white", markersize=6.0, label="Other methods"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
        fontsize=9.2,
    )
    figure.subplots_adjust(left=0.27, right=0.975, top=0.90, bottom=0.13, hspace=0.25, wspace=0.20)
    paths = _save_figure(figure, output_dir, "09_consistency_comparison")
    plt.close(figure)
    return paths


def plot_monte_carlo_figures(matrix: dict, output_dir: Path) -> list[Path]:
    """Generate the paired-ablation and consistency figures in PNG/PDF/SVG."""

    _configure_matplotlib()
    return [
        *plot_paired_ablation_differences(matrix, output_dir),
        *plot_consistency_comparison(matrix, output_dir),
    ]


def plot_generalization_gap(comparison: dict, output_dir: Path) -> list[Path]:
    """Plot paired OOD-minus-ID changes for accuracy and consistency metrics."""

    _configure_matplotlib()
    paired = comparison["paired_gap"]
    methods = tuple(paired.keys())
    positions = np.arange(len(methods))
    figure, axes = plt.subplots(2, 2, figsize=(13.4, 10.8), sharey=True)

    for panel_index, (axis, (metric, xlabel, panel_title)) in enumerate(
        zip(axes.flat, PAIR_METRICS, strict=True)
    ):
        means = np.asarray([paired[method][metric]["mean"] for method in methods])
        lowers = np.asarray([paired[method][metric]["ci_lower"] for method in methods])
        uppers = np.asarray([paired[method][metric]["ci_upper"] for method in methods])
        axis.axvline(0.0, color="#303030", linestyle="--", linewidth=1.0, zorder=1)
        for index, method in enumerate(methods):
            color, marker, marker_size = _method_style(method)
            axis.errorbar(
                means[index],
                positions[index],
                xerr=np.array(
                    [[means[index] - lowers[index]], [uppers[index] - means[index]]]
                ),
                fmt=marker,
                markersize=marker_size,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.7,
                ecolor=color,
                elinewidth=1.3,
                capsize=2.8,
                capthick=1.0,
                alpha=1.0 if method in {
                    "PC-DH-LSTM-AEKF",
                    "GM-9D EKF",
                    "GM-9D EKF + R-CM",
                } else 0.76,
                zorder=3 if method == "PC-DH-LSTM-AEKF" else 2,
            )
        axis.set_xlim(*_padded_limits(lowers, uppers, 0.0))
        _format_delta_axis(axis, np.concatenate((lowers, uppers)))
        axis.set_xlabel(xlabel.replace("difference", "OOD - ID difference"), fontsize=10.5)
        axis.set_title(panel_title, loc="left", fontsize=11.5, fontweight="normal")
        axis.set_ylim(len(methods) - 0.5, -0.5)
        axis.set_yticks(positions)
        if panel_index % 2 == 0:
            axis.set_yticklabels([METHOD_LABELS.get(method, method) for method in methods], fontsize=8.9)
        else:
            axis.tick_params(axis="y", labelleft=False)
        axis.grid(axis="x")
        axis.tick_params(axis="x", labelsize=9)
        axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        f"Paired changes from ID to unseen disturbance family ({comparison['runs']} same-seed trajectories)",
        fontsize=15,
        fontweight="normal",
        y=0.985,
    )
    figure.text(
        0.5,
        0.951,
        (
            f"Difference = {comparison['ood_scenario']} - {comparison['id_scenario']}; "
            "points are means and bars are two-sided 95% t intervals"
        ),
        ha="center",
        va="center",
        fontsize=9.8,
        color="#555555",
    )
    figure.subplots_adjust(
        left=0.29, right=0.975, top=0.91, bottom=0.08, hspace=0.27, wspace=0.20
    )
    paths = _save_figure(figure, output_dir, "10_id_ood_generalization_gap")
    plt.close(figure)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Monte Carlo paper figures")
    parser.add_argument("--input", type=Path, required=True, help="experiment matrix JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("fig"))
    args = parser.parse_args()
    matrix = json.loads(args.input.read_text(encoding="utf-8"))
    for path in plot_monte_carlo_figures(matrix, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
