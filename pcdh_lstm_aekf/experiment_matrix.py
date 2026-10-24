import csv
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.stats import t, ttest_1samp

from .compute_metrics import compute_metrics, compute_segmented_metrics
from .run_adaptive_ekf import run_adaptive_ekf
from .run_adaptive_gm_ekf import run_adaptive_gm_ekf
from .run_classical_aekf import (
    run_coupled_nis_aekf,
    run_innovation_cm_aekf,
    run_sage_husa_fading_aekf,
)
from .run_fixed_ekf import run_fixed_ekf
from .run_gm_ekf import run_gm_9d_ekf
from .run_gm_residual_lstm_ekf import run_gm_residual_lstm_ekf
from .run_known_scale_ekf import run_known_scale_ekf
from .simulate_sequence import simulate_sequence


CORE_METRICS = (
    "position_rss_rmse_m",
    "velocity_rss_rmse_mps",
    "normalized_nis",
    "normalized_nees",
    "nis_95_coverage",
    "nees_95_coverage",
)


def build_method_suite(model, config) -> dict[str, Callable[[dict[str, np.ndarray]], dict]]:
    """Build paired baseline and ablation runners for the Monte Carlo matrix."""

    return {
        "Fixed EKF": lambda sequence: run_fixed_ekf(sequence, config),
        "Innovation-CM AEKF": lambda sequence: run_innovation_cm_aekf(sequence, config),
        "Sage-Husa/Fading AEKF": lambda sequence: run_sage_husa_fading_aekf(sequence, config),
        "Coupled-NIS Q/R AEKF": lambda sequence: run_coupled_nis_aekf(sequence, config),
        "GM-9D EKF": lambda sequence: run_gm_9d_ekf(sequence, config),
        "GM-9D EKF + R-CM": lambda sequence: run_gm_9d_ekf(
            sequence, config, adapt_r=True
        ),
        "Adaptive GM-AEKF": lambda sequence: run_adaptive_gm_ekf(sequence, config),
        "Known-scale EKF": lambda sequence: run_known_scale_ekf(sequence, config),
        "PC-DH-LSTM-AEKF": lambda sequence: run_adaptive_ekf(sequence, model, config),
        "GM-9D EKF + LSTM residual": lambda sequence: run_gm_residual_lstm_ekf(
            sequence,
            model,
            config,
            use_lstm_r_head=False,
            use_nis_feedback=False,
            use_r_cm_correction=True,
        ),
        "GM-9D EKF + LSTM residual (no residual head)": lambda sequence: (
            run_gm_residual_lstm_ekf(
                sequence,
                model,
                config,
                use_residual_head=False,
                use_lstm_r_head=False,
                use_nis_feedback=False,
                use_r_cm_correction=True,
            )
        ),
        "GM-9D EKF + LSTM residual (with LSTM R head)": lambda sequence: (
            run_gm_residual_lstm_ekf(
                sequence,
                model,
                config,
                use_lstm_r_head=True,
                use_nis_feedback=True,
                use_r_cm_correction=True,
            )
        ),
        "Shared NIS Q/R in full model": lambda sequence: run_adaptive_ekf(
            sequence, model, config, use_qr_separation=False
        ),
        "No acceleration head": lambda sequence: run_adaptive_ekf(
            sequence, model, config, use_acceleration_head=False
        ),
        "No R fast channel": lambda sequence: run_adaptive_ekf(
            sequence,
            model,
            config,
            use_r_head=False,
            use_r_fast_channel=False,
            use_r_cm_correction=False,
            use_nis_feedback=False,
        ),
        "No Q slow constraint": lambda sequence: run_adaptive_ekf(
            sequence,
            model,
            config,
            use_q_slow_constraint=False,
            use_q_cm_correction=False,
        ),
        "Deterministic acceleration mean only": lambda sequence: run_adaptive_ekf(
            sequence,
            model,
            config,
            propagate_acceleration_uncertainty=False,
        ),
        "Inference feature-source switch (same weights)": lambda sequence: run_adaptive_ekf(
            sequence, model, config, use_shadow_fixed_features=False
        ),
    }


def summarize_metric_values(values: list[float], confidence: float = 0.95) -> dict[str, float]:
    """Return mean, sample SD, standard error, and two-sided t confidence interval."""

    array = np.asarray(values, dtype=float)
    count = int(array.size)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if count > 1 else 0.0
    standard_error = std / np.sqrt(count) if count else float("nan")
    if count > 1:
        critical = float(t.ppf(0.5 + confidence / 2.0, count - 1))
        margin = critical * standard_error
    else:
        margin = 0.0
    return {
        "n": count,
        "mean": mean,
        "std": std,
        "standard_error": float(standard_error),
        "confidence_level": confidence,
        "ci_lower": mean - margin,
        "ci_upper": mean + margin,
        "ci_half_width": margin,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm family-wise-error adjusted p-values in input order."""

    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running_max = 0.0
    count = values.size
    for rank, index in enumerate(order):
        running_max = max(running_max, (count - rank) * values[index])
        adjusted[index] = min(running_max, 1.0)
    return adjusted.tolist()


def run_monte_carlo_matrix(
    config,
    model,
    *,
    scenario: str,
    runs: int,
    seed_offset: int = 1_000_000,
    show_progress: bool = False,
) -> dict:
    """Run every method over the same independent seeded test trajectories."""

    if runs < 30:
        raise ValueError("Monte Carlo reporting requires at least 30 independent seeds")

    methods = build_method_suite(model, config)
    seeds = [int(config.seed + seed_offset + index) for index in range(runs)]
    per_run = {name: [] for name in methods}
    segmented_per_run = {name: [] for name in methods}
    disturbance_profiles = []

    for run_index, seed in enumerate(seeds, start=1):
        sequence = simulate_sequence(config, seed, scenario)
        profile = sequence["acceleration_profile"]
        disturbance_profiles.append(
            {
                "run": run_index,
                "seed": seed,
                "family": profile["family"],
                "amplitudes_mps2": np.asarray(profile["amplitudes_mps2"]).tolist(),
                "phases_rad": np.asarray(profile["phases_rad"]).tolist(),
                "omega_start_radps": np.asarray(profile["omega_start_radps"]).tolist(),
                "omega_rate_radps2": np.asarray(profile["omega_rate_radps2"]).tolist(),
                "step_times_s": np.asarray(profile["step_times_s"]).tolist(),
                "step_biases_mps2": np.asarray(profile["step_biases_mps2"]).tolist(),
            }
        )
        for method_name, runner in methods.items():
            result = runner(sequence)
            segmented_metrics = compute_segmented_metrics(sequence, result, config)
            metrics = segmented_metrics["full"]
            per_run[method_name].append(
                {
                    "run": run_index,
                    "seed": seed,
                    **{metric: metrics[metric] for metric in CORE_METRICS},
                    "runtime_ms_per_step": metrics["runtime_ms_per_step"],
                    "finite": metrics["finite"],
                }
            )
            segmented_per_run[method_name].append(
                {
                    phase: {metric: values[metric] for metric in CORE_METRICS}
                    for phase, values in segmented_metrics.items()
                }
            )
        if show_progress and (run_index == 1 or run_index % 5 == 0 or run_index == runs):
            print(f"  Monte Carlo progress: {run_index}/{runs} seeds complete")

    summary = {}
    for method_name, rows in per_run.items():
        summary[method_name] = {
            metric: summarize_metric_values([row[metric] for row in rows])
            for metric in CORE_METRICS
        }
        summary[method_name]["runtime_ms_per_step"] = summarize_metric_values(
            [row["runtime_ms_per_step"] for row in rows]
        )
        summary[method_name]["all_finite"] = all(row["finite"] for row in rows)

    reference_name = "PC-DH-LSTM-AEKF"
    paired_difference_vs_full = {}
    reference_rows = per_run[reference_name]
    for method_name, rows in per_run.items():
        if method_name == reference_name:
            continue
        paired_difference_vs_full[method_name] = {
            metric: summarize_metric_values(
                [
                    row[metric] - reference_row[metric]
                    for row, reference_row in zip(rows, reference_rows, strict=True)
                ]
            )
            for metric in CORE_METRICS
        }

    segmented_summary = {}
    for method_name, rows in segmented_per_run.items():
        segmented_summary[method_name] = {
            phase: {
                metric: summarize_metric_values(
                    [row[phase][metric] for row in rows]
                )
                for metric in CORE_METRICS
            }
            for phase in rows[0]
        }

    confirmatory_ablation_names = (
        "Shared NIS Q/R in full model",
        "No acceleration head",
        "No R fast channel",
        "No Q slow constraint",
        "Deterministic acceleration mean only",
    )
    primary_metrics = (
        "position_rss_rmse_m",
        "velocity_rss_rmse_mps",
        "normalized_nis",
        "normalized_nees",
    )
    tests = []
    raw_p_values = []
    for method_name in confirmatory_ablation_names:
        rows = per_run[method_name]
        for metric in primary_metrics:
            differences = np.asarray(
                [
                    row[metric] - reference_row[metric]
                    for row, reference_row in zip(rows, reference_rows, strict=True)
                ],
                dtype=float,
            )
            p_value = float(ttest_1samp(differences, popmean=0.0).pvalue)
            if not np.isfinite(p_value):
                p_value = 1.0
            tests.append(
                {
                    "method": method_name,
                    "metric": metric,
                    "mean_paired_difference": float(np.mean(differences)),
                    "raw_p_value": p_value,
                }
            )
            raw_p_values.append(p_value)
    for test, adjusted in zip(tests, holm_adjust(raw_p_values), strict=True):
        test["holm_adjusted_p_value"] = adjusted
        test["reject_at_0_05"] = bool(adjusted < 0.05)

    return {
        "scenario": scenario,
        "runs": runs,
        "seed_offset": int(seed_offset),
        "seed_policy": f"config.seed + {seed_offset} + run_index_zero_based",
        "seeds": seeds,
        "disturbance_profiles": disturbance_profiles,
        "confidence_interval": "two-sided 95% Student-t interval across seeds",
        "core_metrics": CORE_METRICS,
        "per_run": per_run,
        "segmented_per_run": segmented_per_run,
        "summary": summary,
        "segmented_summary": segmented_summary,
        "paired_difference_definition": "method minus PC-DH-LSTM-AEKF on the same seed",
        "paired_difference_vs_full": paired_difference_vs_full,
        "confirmatory_ablation_tests": {
            "family": "5 ablations x 4 primary metrics",
            "correction": "Holm family-wise error correction",
            "alpha": 0.05,
            "tests": tests,
            "excluded_exploratory_diagnostic": (
                "Inference feature-source switch (same weights)"
            ),
        },
    }


def save_experiment_matrix(
    matrix: dict,
    json_path: Path,
    csv_path: Path,
    per_run_csv_path: Path | None = None,
) -> None:
    """Persist full JSON, summary CSV, and optional seed-level CSV."""

    json_path.write_text(json.dumps(matrix, ensure_ascii=False, indent=2), encoding="utf-8")

    statistics = ("mean", "std", "ci_lower", "ci_upper")
    fieldnames = ["method", "runs"]
    for metric in CORE_METRICS:
        fieldnames.extend(f"{metric}_{statistic}" for statistic in statistics)
    fieldnames.extend(
        f"runtime_ms_per_step_{statistic}" for statistic in statistics
    )
    fieldnames.append("all_finite")

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method_name, values in matrix["summary"].items():
            row = {
                "method": method_name,
                "runs": matrix["runs"],
                "all_finite": values["all_finite"],
            }
            for metric in CORE_METRICS:
                for statistic in statistics:
                    row[f"{metric}_{statistic}"] = values[metric][statistic]
            for statistic in statistics:
                row[f"runtime_ms_per_step_{statistic}"] = values[
                    "runtime_ms_per_step"
                ][statistic]
            writer.writerow(row)

    if per_run_csv_path is not None:
        run_fields = ["method", "run", "seed", *CORE_METRICS, "runtime_ms_per_step", "finite"]
        with per_run_csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=run_fields)
            writer.writeheader()
            for method_name, rows in matrix["per_run"].items():
                for values in rows:
                    writer.writerow({"method": method_name, **values})


def build_paired_scenario_comparison(id_matrix: dict, ood_matrix: dict) -> dict:
    """Build same-seed OOD-minus-ID statistics for every method and metric."""

    if id_matrix["runs"] != ood_matrix["runs"] or id_matrix["seeds"] != ood_matrix["seeds"]:
        raise ValueError("ID and OOD matrices must contain the same ordered seeds")
    if tuple(id_matrix["per_run"]) != tuple(ood_matrix["per_run"]):
        raise ValueError("ID and OOD matrices must contain the same ordered methods")

    paired_gap = {}
    per_run_gap = {}
    for method_name in id_matrix["per_run"]:
        id_rows = id_matrix["per_run"][method_name]
        ood_rows = ood_matrix["per_run"][method_name]
        method_rows = []
        for id_row, ood_row in zip(id_rows, ood_rows, strict=True):
            if id_row["seed"] != ood_row["seed"]:
                raise ValueError("ID/OOD row seed mismatch")
            method_rows.append(
                {
                    "run": id_row["run"],
                    "seed": id_row["seed"],
                    **{
                        metric: ood_row[metric] - id_row[metric]
                        for metric in CORE_METRICS
                    },
                }
            )
        per_run_gap[method_name] = method_rows
        paired_gap[method_name] = {
            metric: summarize_metric_values([row[metric] for row in method_rows])
            for metric in CORE_METRICS
        }

    return {
        "id_scenario": id_matrix["scenario"],
        "ood_scenario": ood_matrix["scenario"],
        "runs": id_matrix["runs"],
        "seeds": id_matrix["seeds"],
        "paired_difference_definition": "OOD scenario minus ID scenario on the same seed",
        "core_metrics": CORE_METRICS,
        "id_summary": id_matrix["summary"],
        "ood_summary": ood_matrix["summary"],
        "per_run_gap": per_run_gap,
        "paired_gap": paired_gap,
    }


def save_paired_scenario_comparison(
    comparison: dict,
    json_path: Path,
    csv_path: Path,
) -> None:
    """Save the paired OOD-minus-ID generalization statistics."""

    json_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    statistics = ("mean", "std", "ci_lower", "ci_upper")
    fieldnames = ["method", "runs"]
    for metric in CORE_METRICS:
        fieldnames.extend(f"{metric}_{statistic}" for statistic in statistics)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method_name, metric_values in comparison["paired_gap"].items():
            row = {"method": method_name, "runs": comparison["runs"]}
            for metric in CORE_METRICS:
                for statistic in statistics:
                    row[f"{metric}_{statistic}"] = metric_values[metric][statistic]
            writer.writerow(row)


def format_matrix_summary(matrix: dict) -> str:
    """Format mean, SD, and 95% confidence intervals for console output."""

    lines = [
        "  Method                         Position RMSE mean+/-SD [95% CI]    "
        "NIS/DOF mean+/-SD [95% CI]    NEES/6 mean+/-SD [95% CI]"
    ]
    for method_name, values in matrix["summary"].items():
        position = values["position_rss_rmse_m"]
        nis = values["normalized_nis"]
        nees = values["normalized_nees"]
        lines.append(
            f"  {method_name:30s} "
            f"{position['mean']:.4f}+/-{position['std']:.4f} "
            f"[{position['ci_lower']:.4f}, {position['ci_upper']:.4f}]    "
            f"{nis['mean']:.3f}+/-{nis['std']:.3f} "
            f"[{nis['ci_lower']:.3f}, {nis['ci_upper']:.3f}]    "
            f"{nees['mean']:.3f}+/-{nees['std']:.3f} "
            f"[{nees['ci_lower']:.3f}, {nees['ci_upper']:.3f}]"
        )
    return "\n".join(lines)
