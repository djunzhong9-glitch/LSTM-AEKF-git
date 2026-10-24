import argparse
import json
from pathlib import Path

from pcdh_lstm_aekf.experiment_matrix import summarize_metric_values


METHODS = (
    "Fixed EKF",
    "Innovation-CM AEKF",
    "Sage-Husa/Fading AEKF",
    "Coupled-NIS Q/R AEKF",
    "GM-9D EKF",
    "GM-9D EKF + R-CM",
    "Known-scale EKF",
    "PC-DH-LSTM-AEKF",
    "GM-9D EKF + LSTM residual",
)
METRICS = (
    "position_rss_rmse_m",
    "velocity_rss_rmse_mps",
    "normalized_nis",
    "normalized_nees",
)
METRIC_LABELS = {
    "position_rss_rmse_m": "位置 RMSE (m)",
    "velocity_rss_rmse_mps": "速度 RMSE (m/s)",
    "normalized_nis": "NIS/DOF",
    "normalized_nees": "NEES/6",
    "nis_95_coverage": "NIS 95% 覆盖率",
    "nees_95_coverage": "NEES 95% 覆盖率",
}
METRIC_DIGITS = {
    "position_rss_rmse_m": 6,
    "velocity_rss_rmse_mps": 6,
    "normalized_nis": 4,
    "normalized_nees": 3,
    "nis_95_coverage": 4,
    "nees_95_coverage": 4,
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def phase_summary(data: dict, method: str, phase: str) -> dict:
    values = data["summary"][method]
    if "across_training_seeds" in values:
        return values["across_training_seeds"][phase]
    return values[phase]


def format_stat(statistics: dict, metric: str) -> str:
    digits = METRIC_DIGITS[metric]
    return (
        f"{statistics['mean']:.{digits}f} ± {statistics['std']:.{digits}f} "
        f"[{statistics['ci_lower']:.{digits}f}, "
        f"{statistics['ci_upper']:.{digits}f}]"
    )


def row_lookup(data: dict, method: str) -> dict:
    return {
        (row["training_seed"], row["test_seed"]): row
        for row in data["rows_by_method"][method]
    }


def paired_scenario_summary(
    in_distribution: dict,
    out_of_distribution: dict,
    method: str,
    metric: str,
) -> dict:
    id_rows = row_lookup(in_distribution, method)
    ood_rows = row_lookup(out_of_distribution, method)
    training_seeds = sorted(
        seed for seed, _ in id_rows if seed is not None
    )
    if not training_seeds:
        differences = [
            ood_rows[key]["metrics"]["post_burn_in"][metric]
            - row["metrics"]["post_burn_in"][metric]
            for key, row in id_rows.items()
        ]
        return summarize_metric_values(differences)
    seed_differences = []
    for training_seed in sorted(set(training_seeds)):
        differences = [
            ood_rows[key]["metrics"]["post_burn_in"][metric]
            - row["metrics"]["post_burn_in"][metric]
            for key, row in id_rows.items()
            if key[0] == training_seed
        ]
        seed_differences.append(sum(differences) / len(differences))
    return summarize_metric_values(seed_differences)


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" if i == 0 else "---:" for i in range(len(headers))) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def build_report(root: Path) -> str:
    training = read_json(root / "training_ensemble.json")
    gm = read_json(root / "gm_validation_tuning.json")
    id_data = read_json(root / "crossed_id.json")
    factors = read_json(root / "single_factor_scan.json")

    lines = [
        "# Major Revision v4 正式实验结果",
        "",
        "## 实验身份与统计口径",
        "",
        "- 版本：`major-revision-v4`。",
        "- 训练：固定 320 条数据与 256/64 划分，5 个独立初始化和批次顺序 seed。",
        "- 测试：分布内结构化非平稳场景和各单因素水平分别使用 30 个共同 test seed。",
        "- 主时段：25 s burn-in 后；另保存 full、early、middle、late。",
        "- 学习方法的训练级 CI：以 5 个训练 seed 各自的 30-seed 均值为推断单位。",
        "- 传统方法的 CI：以 30 个 test seed 为推断单位。",
        "- 表格格式：均值 ± 样本标准差 [双侧 95% Student-t CI]。",
        "",
        "## 五次独立训练",
        "",
    ]
    training_rows = []
    for record in training["models"]:
        history = record["history"]
        best_epoch = history["best_epoch"]
        training_rows.append(
            [
                str(record["training_seed"]),
                "NA（中断前曲线未持久化）" if best_epoch is None else str(best_epoch),
                f"{history['best_validation_loss']:.8f}",
                "是" if history.get("training_curve_available", True) else "否",
                f"`{record['sha256']}`",
            ]
        )
    lines.extend(
        markdown_table(
            ["训练 seed", "最佳 epoch", "最佳验证损失", "逐 epoch 曲线", "权重 SHA-256"],
            training_rows,
        )
    )
    lines.extend(
        [
            "",
            "前四个完整最佳权重在长时运行管道中断前已写盘，恢复时重新核验验证损失、固定划分、加速度残差协方差与 SHA-256；其逐 epoch 曲线未被追溯性补造。第五个实例保留完整曲线。",
            "",
            "## 每个训练实例内部的 30-seed Monte Carlo",
            "",
        ]
    )
    per_seed = id_data["summary"]["GM-9D EKF + LSTM residual"][
        "per_training_seed_test_statistics"
    ]
    seed_rows = []
    for training_seed in training["training_seeds"]:
        summary = per_seed[str(training_seed)]["post_burn_in"]
        seed_rows.append(
            [str(training_seed)]
            + [
                format_stat(summary[metric], metric)
                for metric in (
                    "position_rss_rmse_m",
                    "velocity_rss_rmse_mps",
                    "normalized_nis",
                    "normalized_nees",
                    "nis_95_coverage",
                    "nees_95_coverage",
                )
            ]
        )
    lines.extend(
        markdown_table(
            [
                "训练 seed",
                "位置 RMSE",
                "速度 RMSE",
                "NIS/DOF",
                "NEES/6",
                "NIS 覆盖率",
                "NEES 覆盖率",
            ],
            seed_rows,
        )
    )
    lines.append("")
    lines.extend(
        [
            "## GM-9D 验证集冻结参数",
            "",
            f"- 冻结参数：`tau={gm['selected_tau_s']} s`，`q_g={gm['selected_drive_psd_m2_s5']:.6g} m^2/s^5`。",
            f"- 原始最优：`tau={gm['grid_diagnostics']['raw_best_tau_s']} s`，`q_g={gm['grid_diagnostics']['raw_best_drive_psd_m2_s5']:.6g}`。",
            f"- 1% 近优候选 {gm['grid_diagnostics']['near_optimal_candidate_count']} 个，其中双轴内部候选 {gm['grid_diagnostics']['interior_near_optimal_candidate_count']} 个。",
            f"- 网格状态：`{gm['grid_diagnostics']['drive_grid_status']}`；测试集参与选择：`{gm['test_set_used_for_selection']}`。",
            "",
            "## ID 主结果（post-burn-in）",
            "",
        ]
    )
    accuracy_rows = []
    consistency_rows = []
    for method in METHODS:
        summary = phase_summary(id_data, method, "post_burn_in")
        accuracy_rows.append(
            [
                method,
                format_stat(summary["position_rss_rmse_m"], "position_rss_rmse_m"),
                format_stat(summary["velocity_rss_rmse_mps"], "velocity_rss_rmse_mps"),
            ]
        )
        consistency_rows.append(
            [
                method,
                format_stat(summary["normalized_nis"], "normalized_nis"),
                format_stat(summary["normalized_nees"], "normalized_nees"),
                format_stat(summary["nis_95_coverage"], "nis_95_coverage"),
                format_stat(summary["nees_95_coverage"], "nees_95_coverage"),
            ]
        )
    lines.extend(markdown_table(["方法", "位置 RMSE", "速度 RMSE"], accuracy_rows))
    lines.extend(["", "### ID 一致性与覆盖率", ""])
    lines.extend(
        markdown_table(
            ["方法", "NIS/DOF", "NEES/6", "NIS 95% 覆盖率", "NEES 95% 覆盖率"],
            consistency_rows,
        )
    )

    lines.extend(["", "## 确认性消融：消融 - 完整方法", ""])
    ablation_rows = []
    for test in id_data["confirmatory_ablation_tests"]["tests"]:
        difference = test["difference_summary"]
        ablation_rows.append(
            [
                test["method"],
                METRIC_LABELS[test["metric"]],
                format_stat(difference, test["metric"]),
                f"{test['holm_adjusted_p_value']:.4g}",
                "是" if test["reject_at_0_05"] else "否",
            ]
        )
    lines.extend(
        markdown_table(
            ["消融", "指标", "配对差", "Holm p", "0.05 拒绝"],
            ablation_rows,
        )
    )

    lines.extend(["", "## 单因素边界：GM-9D - 完整方法", ""])
    factor_rows = []
    for factor, levels in factors["gm_minus_learning_crossover"].items():
        for level, metrics in levels.items():
            position = metrics["position_rss_rmse_m"]
            velocity = metrics["velocity_rss_rmse_mps"]
            factor_rows.append(
                [
                    factor,
                    level,
                    format_stat(position["summary"], "position_rss_rmse_m"),
                    position["ordering_at_95_percent"],
                    format_stat(velocity["summary"], "velocity_rss_rmse_mps"),
                    velocity["ordering_at_95_percent"],
                ]
            )
    lines.extend(
        markdown_table(
            ["因子", "水平", "位置差", "位置排序", "速度差", "速度排序"],
            factor_rows,
        )
    )

    lines.extend(["", "## 分阶段结果", ""])
    stage_rows = []
    for method in ("GM-9D EKF", "PC-DH-LSTM-AEKF"):
        for phase in ("full", "post_burn_in", "early", "middle", "late"):
            summary = phase_summary(id_data, method, phase)
            stage_rows.append(
                [method, phase]
                + [format_stat(summary[metric], metric) for metric in METRICS]
            )
    lines.extend(
        markdown_table(
            ["方法", "阶段"] + [METRIC_LABELS[metric] for metric in METRICS],
            stage_rows,
        )
    )

    lines.extend(
        [
            "",
            "## 结果解释边界",
            "",
            "- 统计显著不等于工程效应大；共享 NIS Q/R 与 Q 慢约束的差异必须结合效应量解释。",
            "- R 快速通道的主要作用由 NIS/覆盖率衡量，不能只看位置 RMSE。",
            "- 常值验证残差协方差能缓解但不能保证状态一致性，仍应诚实报告 NEES/6 与覆盖率。",
            "- 本实验不主张跨扰动族 OOD 泛化能力；有效边界以分布内结构化非平稳场景和 RMS 控制单因素扫描为准。",
            "- 五次训练共享固定数据与划分，区间不覆盖训练数据重采样、高保真摄动或真实飞行数据不确定性。",
            "",
            "## 主图",
            "",
            "- `fig/11_single_factor_boundaries.pdf`：四类单因素边界。",
            "- `fig/12_training_seed_summary.pdf`：五训练 seed 的分布内结果。",
            "- `fig/13_holm_ablation_training_seeds.pdf`：确认性消融配对差与 95% CI。",
            "- `fig/14_consistency_coverage.pdf`：NIS、NEES 与两类覆盖率。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Chinese formal result report")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts") / "major_revision_v4",
    )
    parser.add_argument("--output", type=Path, default=Path("MONTE_CARLO_RESULTS.md"))
    args = parser.parse_args()
    args.output.write_text(
        build_report(args.artifact_dir.resolve()),
        encoding="utf-8",
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
