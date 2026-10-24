from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .plot_monte_carlo import _configure_matplotlib, _save_figure


METHOD_STYLES = {
    "Fixed EKF": ("#7A8793", "o"),
    "GM-9D EKF": ("#356CB6", "P"),
    "Adaptive GM-AEKF": ("#8A5CC2", "^"),
    "PC-DH-LSTM-AEKF": ("#D44E52", "D"),
    "GM-9D EKF + LSTM residual": ("#2B9A8A", "s"),
}
FACTOR_X_LABELS = {
    "amplitude_scale": "Disturbance amplitude scale",
    "frequency_scale": "Disturbance frequency scale (RMS matched)",
    "transition_duration_s": "Step rise time (s, RMS matched)",
    "measurement_interval_s": "Measurement interval (s)",
}
MAIN_FACTOR_BOUNDARY_PANELS = (
    ("amplitude_scale", "(a) Amplitude scan"),
    ("transition_duration_s", "(b) Abruptness scan"),
)
FULL_FACTOR_BOUNDARY_PANELS = (
    ("amplitude_scale", "(a) Amplitude scan"),
    ("frequency_scale", "(b) Frequency scan"),
    ("transition_duration_s", "(c) Abruptness scan"),
    ("measurement_interval_s", "(d) Measurement-interval scan"),
)
METRIC_LABELS = {
    "position_rss_rmse_m": "Position RSS RMSE (m)",
    "velocity_rss_rmse_mps": "Velocity RSS RMSE (m/s)",
    "normalized_nis": "NIS/DOF",
    "normalized_nees": "NEES/6",
}


def _scan_statistics(factor_data: dict, factor: str, level: float, method: str, metric: str):
    values = factor_data["summary"][factor][str(level)][method]
    if "across_training_seeds" in values:
        values = values["across_training_seeds"]
    statistics = values["post_burn_in"][metric]
    return statistics["mean"], statistics["ci_lower"], statistics["ci_upper"]


def _draw_factor_panel(axis, factor_data: dict, factor: str, title: str) -> None:
    specification = factor_data["factor_specs"][factor]
    levels = np.asarray(specification["levels"], dtype=float)
    for method, (color, marker) in METHOD_STYLES.items():
        statistics = [
            _scan_statistics(
                factor_data,
                factor,
                float(level),
                method,
                "position_rss_rmse_m",
            )
            for level in levels
        ]
        means = np.asarray([item[0] for item in statistics])
        lower = np.asarray([item[1] for item in statistics])
        upper = np.asarray([item[2] for item in statistics])
        axis.plot(levels, means, color=color, marker=marker, linewidth=1.6, label=method)
        axis.fill_between(levels, lower, upper, color=color, alpha=0.14, linewidth=0)
    axis.set_title(title, loc="left", fontsize=10.5)
    axis.set_xlabel(FACTOR_X_LABELS[factor])
    axis.set_ylabel("Post-burn-in position RSS RMSE (m)")
    axis.grid(True, axis="both")
    if factor in {"amplitude_scale", "frequency_scale"}:
        axis.axvspan(0.65, 1.35, color="#59A14F", alpha=0.08)
    elif factor == "transition_duration_s":
        axis.axvline(0.0, color="#59A14F", linestyle="--", linewidth=0.9)
    else:
        axis.axvline(0.1, color="#59A14F", linestyle="--", linewidth=0.9)


def plot_factor_boundaries(factor_data: dict, output_dir: Path) -> list[Path]:
    _configure_matplotlib()
    paths: list[Path] = []

    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), constrained_layout=True)
    for axis, (factor, title) in zip(axes.flat, MAIN_FACTOR_BOUNDARY_PANELS):
        _draw_factor_panel(axis, factor_data, factor, title)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8.5)
    figure.suptitle(
        "Single-factor performance boundaries: 5 fixed training instances x 30 shared test seeds",
        fontsize=13,
    )
    paths.extend(_save_figure(figure, output_dir, "11_single_factor_boundaries"))

    figure, axes = plt.subplots(2, 2, figsize=(11.6, 8.0), constrained_layout=True)
    for axis, (factor, title) in zip(axes.flat, FULL_FACTOR_BOUNDARY_PANELS):
        _draw_factor_panel(axis, factor_data, factor, title)
    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=8.5)
    figure.suptitle(
        "Full single-factor performance boundaries: 5 fixed training instances x 30 shared test seeds",
        fontsize=13,
    )
    paths.extend(_save_figure(figure, output_dir, "A1_single_factor_boundaries_full"))
    return paths


def _crossed_statistics(data: dict, method: str, metric: str):
    values = data["summary"][method]
    if "across_training_seeds" in values:
        values = values["across_training_seeds"]
    return values["post_burn_in"][metric]


def plot_training_seed_summary(
    id_data: dict,
    output_dir: Path,
) -> list[Path]:
    _configure_matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.6), constrained_layout=True)
    methods = (
        "Fixed EKF",
        "GM-9D EKF",
        "Adaptive GM-AEKF",
        "PC-DH-LSTM-AEKF",
        "GM-9D EKF + LSTM residual",
    )
    for axis_index, (axis, metric) in enumerate(zip(axes.flat, METRIC_LABELS)):
        x = np.arange(len(methods), dtype=float)
        statistics = [_crossed_statistics(id_data, method, metric) for method in methods]
        means = np.asarray([item["mean"] for item in statistics])
        errors = np.vstack(
            (
                means - np.asarray([item["ci_lower"] for item in statistics]),
                np.asarray([item["ci_upper"] for item in statistics]) - means,
            )
        )
        axis.errorbar(
            x,
            means,
            yerr=errors,
            fmt="o",
            capsize=3,
            linewidth=1.2,
            color="#356CB6",
        )
        axis.set_xticks(
            x,
            ("Fixed", "GM-9D", "AGM", "PC-DH", "GM+Res"),
            rotation=12,
        )
        axis.set_ylabel(METRIC_LABELS[metric])
        axis.grid(True, axis="y")
        if metric in {"normalized_nis", "normalized_nees"}:
            axis.axhline(1.0, color="#555555", linestyle="--", linewidth=0.9)
    figure.suptitle(
        "In-distribution results across training seeds (post-burn-in, 95% CI)",
        fontsize=13,
    )
    return _save_figure(figure, output_dir, "12_training_seed_summary")


def plot_holm_ablation(id_data: dict, output_dir: Path) -> list[Path]:
    _configure_matplotlib()
    tests = id_data["confirmatory_ablation_tests"]["tests"]
    methods = list(dict.fromkeys(test["method"] for test in tests))
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), constrained_layout=True)
    for axis_index, (axis, metric) in enumerate(zip(axes.flat, METRIC_LABELS)):
        selected = [test for test in tests if test["metric"] == metric]
        means = np.asarray([test["difference_summary"]["mean"] for test in selected])
        lower = np.asarray([test["difference_summary"]["ci_lower"] for test in selected])
        upper = np.asarray([test["difference_summary"]["ci_upper"] for test in selected])
        y = np.arange(len(methods))
        colors = ["#D44E52" if test["reject_at_0_05"] else "#7A8793" for test in selected]
        for index, test in enumerate(selected):
            axis.errorbar(
                means[index],
                y[index],
                xerr=[[means[index] - lower[index]], [upper[index] - means[index]]],
                fmt="o",
                color=colors[index],
                capsize=3,
            )
        axis.axvline(0.0, color="#555555", linestyle="--", linewidth=0.9)
        axis.set_yticks(y, methods if axis_index % 2 == 0 else [""] * len(methods))
        axis.set_xlabel(f"Ablation minus full model: {METRIC_LABELS[metric]}")
        axis.grid(True, axis="x")
    figure.suptitle(
        "Paired ablation differences across 5 training seeds; red denotes Holm-adjusted p < 0.05",
        fontsize=12.5,
    )
    return _save_figure(figure, output_dir, "13_holm_ablation_training_seeds")


def plot_adaptive_gm_ablation(id_data: dict, output_dir: Path) -> list[Path]:
    _configure_matplotlib()
    tests = id_data["adaptive_gm_ablation_tests"]["tests"]
    methods = list(dict.fromkeys(test["method"] for test in tests))
    labels = [
        method.replace("Adaptive GM-AEKF ", "").strip("()")
        if method.startswith("Adaptive GM-AEKF")
        else method
        for method in methods
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11.4, 7.8), constrained_layout=True)
    for axis_index, (axis, metric) in enumerate(zip(axes.flat, METRIC_LABELS)):
        selected = [test for test in tests if test["metric"] == metric]
        means = np.asarray([test["difference_summary"]["mean"] for test in selected])
        lower = np.asarray([test["difference_summary"]["ci_lower"] for test in selected])
        upper = np.asarray([test["difference_summary"]["ci_upper"] for test in selected])
        y = np.arange(len(methods))
        colors = ["#8A5CC2" if test["reject_at_0_05"] else "#7A8793" for test in selected]
        for index, test in enumerate(selected):
            axis.errorbar(
                means[index],
                y[index],
                xerr=[[means[index] - lower[index]], [upper[index] - means[index]]],
                fmt="o",
                color=colors[index],
                capsize=3,
            )
        axis.axvline(0.0, color="#555555", linestyle="--", linewidth=0.9)
        axis.set_yticks(y, labels if axis_index % 2 == 0 else [""] * len(methods))
        axis.set_xlabel(f"Variant minus Adaptive GM-AEKF: {METRIC_LABELS[metric]}")
        axis.grid(True, axis="x")
    figure.suptitle(
        "Adaptive GM-AEKF paired ablation differences; purple denotes Holm-adjusted p < 0.05",
        fontsize=12.5,
    )
    return _save_figure(figure, output_dir, "15_adaptive_gm_ablation")


def plot_consistency_coverage(id_data: dict, output_dir: Path) -> list[Path]:
    _configure_matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), constrained_layout=True)
    methods = (
        "Fixed EKF",
        "Innovation-CM AEKF",
        "Coupled-NIS Q/R AEKF",
        "GM-9D EKF",
        "GM-9D EKF + R-CM",
        "Adaptive GM-AEKF",
        "PC-DH-LSTM-AEKF",
        "GM-9D EKF + LSTM residual",
        "No R fast channel",
    )
    labels = (
        "Fixed",
        "R-CM",
        "Coupled",
        "GM-9D",
        "GM+R",
        "AGM",
        "PC-DH",
        "GM+Res",
        "No R-fast",
    )
    metrics = (
        "normalized_nis",
        "normalized_nees",
        "nis_95_coverage",
        "nees_95_coverage",
    )
    panel_labels = ("(a) NIS/DOF", "(b) NEES/6", "(c) NIS 95% coverage", "(d) NEES 95% coverage")
    x = np.arange(len(methods), dtype=float)
    for axis, metric, panel_label in zip(axes.flat, metrics, panel_labels):
        statistics = [_crossed_statistics(id_data, method, metric) for method in methods]
        means = np.asarray([item["mean"] for item in statistics])
        errors = np.vstack(
            (
                means - np.asarray([item["ci_lower"] for item in statistics]),
                np.asarray([item["ci_upper"] for item in statistics]) - means,
            )
        )
        colors = [
            "#D44E52"
            if method == "PC-DH-LSTM-AEKF"
            else "#2B9A8A"
            if method == "GM-9D EKF + LSTM residual"
            else "#8A5CC2"
            if method == "Adaptive GM-AEKF"
            else "#527B9E"
            for method in methods
        ]
        for index in range(len(methods)):
            axis.errorbar(
                x[index],
                means[index],
                yerr=[[errors[0, index]], [errors[1, index]]],
                fmt="o",
                color=colors[index],
                capsize=3,
                linewidth=1.2,
            )
        axis.axhline(
            1.0 if metric in {"normalized_nis", "normalized_nees"} else 0.95,
            color="#555555",
            linestyle="--",
            linewidth=0.9,
        )
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.set_title(panel_label, loc="left", fontsize=10.5)
        axis.set_ylabel(METRIC_LABELS.get(metric, "95% coverage"))
        axis.grid(True, axis="y")
    figure.suptitle("ID consistency and coverage (post-burn-in, 95% CI)", fontsize=13)
    return _save_figure(figure, output_dir, "14_consistency_coverage")
