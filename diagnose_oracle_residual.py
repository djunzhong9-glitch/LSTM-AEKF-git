import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pcdh_lstm_aekf.compute_metrics import compute_metrics
from pcdh_lstm_aekf.make_config import make_config
from pcdh_lstm_aekf.run_adaptive_ekf import precompute_lstm_outputs
from pcdh_lstm_aekf.run_fixed_ekf import run_fixed_ekf
from pcdh_lstm_aekf.run_gm_ekf import run_gm_9d_ekf
from pcdh_lstm_aekf.run_gm_residual_lstm_ekf import run_gm_residual_lstm_ekf
from pcdh_lstm_aekf.save_model import load_model
from pcdh_lstm_aekf.simulate_sequence import simulate_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose GM-Res LSTM residual learning against an oracle residual."
    )
    parser.add_argument("--mode", choices=("quick", "paper"), default="paper")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument(
        "--scenarios",
        default="nonlinear_structured",
        help="Comma-separated scenarios to diagnose.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts") / "major_revision_v4",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "oracle_residual_diagnostics",
    )
    return parser.parse_args()


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 4 or float(np.std(x)) < 1e-15 or float(np.std(y)) < 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _best_lag_correlation(
    estimate: np.ndarray,
    target: np.ndarray,
    *,
    max_lag: int = 5,
) -> dict[str, float | int]:
    best = {"lag": 0, "correlation": float("nan")}
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            estimate_slice = estimate[:lag]
            target_slice = target[-lag:]
        elif lag > 0:
            estimate_slice = estimate[lag:]
            target_slice = target[:-lag]
        else:
            estimate_slice = estimate
            target_slice = target
        correlation = _corrcoef(estimate_slice, target_slice)
        if not np.isfinite(best["correlation"]) or (
            np.isfinite(correlation)
            and abs(correlation) > abs(float(best["correlation"]))
        ):
            best = {"lag": int(lag), "correlation": float(correlation)}
    return best


def _post_burn_slice(config, steps: int) -> slice:
    burn = min(int(round(config.burn_in_seconds / config.dt)), max(0, steps - 3))
    return slice(burn, steps)


def _metric_row(
    scenario: str,
    method: str,
    test_seed: int,
    training_seed: int | None,
    sequence,
    result,
    config,
) -> dict[str, object]:
    metrics = compute_metrics(
        sequence,
        result,
        start_index=_post_burn_slice(config, int(sequence["truth"].shape[0])).start,
    )
    return {
        "scenario": scenario,
        "method": method,
        "test_seed": int(test_seed),
        "training_seed": training_seed,
        "position_rss_rmse_m": metrics["position_rss_rmse_m"],
        "velocity_rss_rmse_mps": metrics["velocity_rss_rmse_mps"],
        "normalized_nis": metrics["normalized_nis"],
        "normalized_nees": metrics["normalized_nees"],
        "nis_95_coverage": metrics["nis_95_coverage"],
        "nees_95_coverage": metrics["nees_95_coverage"],
        "runtime_ms_per_step": metrics["runtime_ms_per_step"],
    }


def _diagnostic_row(
    scenario: str,
    test_seed: int,
    training_seed: int,
    sequence,
    result,
    lstm_outputs,
    config,
) -> dict[str, object]:
    burn = _post_burn_slice(config, int(sequence["truth"].shape[0]))
    true_acceleration = np.asarray(sequence["acceleration"], dtype=float)
    learned_total = np.asarray(lstm_outputs["acceleration"], dtype=float)
    gm_after_update = np.asarray(result["gm_acceleration_estimate"], dtype=float)
    injected_residual = np.asarray(result["residual_acceleration_estimate"], dtype=float)
    same_time_target_residual = true_acceleration - gm_after_update
    same_time_learned_residual = learned_total - gm_after_update
    interval_target_residual = same_time_target_residual[:-1]
    interval_injected_residual = injected_residual[1:]

    total_corr_by_axis = [
        _corrcoef(learned_total[burn, axis], true_acceleration[burn, axis])
        for axis in range(3)
    ]
    same_time_corr_by_axis = [
        _corrcoef(
            same_time_learned_residual[burn, axis],
            same_time_target_residual[burn, axis],
        )
        for axis in range(3)
    ]
    shifted = slice(max(burn.start - 1, 0), burn.stop - 1)
    interval_corr_by_axis = [
        _corrcoef(
            interval_injected_residual[shifted, axis],
            interval_target_residual[shifted, axis],
        )
        for axis in range(3)
    ]
    best_lags = [
        _best_lag_correlation(
            same_time_learned_residual[burn, axis],
            same_time_target_residual[burn, axis],
        )
        for axis in range(3)
    ]
    proposal = same_time_learned_residual[burn]
    clipped = np.abs(proposal) >= (np.asarray(config.a_max, dtype=float) * 0.999)

    def mse(a, b) -> float:
        return float(np.mean(np.sum(np.square(np.asarray(a) - np.asarray(b)), axis=1)))

    target = same_time_target_residual[burn]
    estimate = same_time_learned_residual[burn]
    return {
        "scenario": scenario,
        "test_seed": int(test_seed),
        "training_seed": int(training_seed),
        "total_acceleration_corr_mean": float(np.nanmean(total_corr_by_axis)),
        "total_acceleration_corr_xyz": total_corr_by_axis,
        "same_time_residual_corr_mean": float(np.nanmean(same_time_corr_by_axis)),
        "same_time_residual_corr_xyz": same_time_corr_by_axis,
        "interval_injected_residual_corr_mean": float(np.nanmean(interval_corr_by_axis)),
        "interval_injected_residual_corr_xyz": interval_corr_by_axis,
        "best_lag_steps_xyz": [item["lag"] for item in best_lags],
        "best_lag_corr_xyz": [item["correlation"] for item in best_lags],
        "learned_residual_mse": mse(estimate, target),
        "zero_residual_mse": mse(np.zeros_like(target), target),
        "learned_to_zero_mse_ratio": mse(estimate, target)
        / max(mse(np.zeros_like(target), target), 1e-30),
        "clip_fraction": float(np.mean(clipped)),
        "true_residual_rms_mps2": float(np.sqrt(np.mean(np.sum(np.square(target), axis=1)))),
        "learned_residual_rms_mps2": float(
            np.sqrt(np.mean(np.sum(np.square(estimate), axis=1)))
        ),
    }


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    artifact_dir = (project_dir / args.artifact_dir).resolve()
    output_dir = (project_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = make_config(args.mode)
    gm_path = artifact_dir / "gm_validation_tuning.json"
    ensemble_path = artifact_dir / "training_ensemble.json"
    gm_tuning = json.loads(gm_path.read_text(encoding="utf-8"))
    config.gm_tau_s = float(gm_tuning["selected_tau_s"])
    config.gm_drive_psd = float(gm_tuning["selected_drive_psd_m2_s5"])

    ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
    models = [
        (int(record["training_seed"]), load_model(Path(record["model_path"])))
        for record in ensemble["models"]
    ]

    scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]
    metric_rows = []
    diagnostic_rows = []
    for scenario in scenarios:
        print(f"Scenario {scenario}: {args.runs} test seeds")
        for index in range(args.runs):
            test_seed = int(config.seed + 2_000_000 + index)
            sequence = simulate_sequence(config, test_seed, scenario)
            fixed_result = run_fixed_ekf(sequence, config)
            gm_result = run_gm_9d_ekf(sequence, config)
            gm_r_result = run_gm_9d_ekf(sequence, config, adapt_r=True)
            metric_rows.append(
                _metric_row(scenario, "GM-9D EKF", test_seed, None, sequence, gm_result, config)
            )
            metric_rows.append(
                _metric_row(
                    scenario,
                    "GM-9D EKF + R-CM",
                    test_seed,
                    None,
                    sequence,
                    gm_r_result,
                    config,
                )
            )

            oracle_outputs = {
                "acceleration": np.asarray(sequence["acceleration"], dtype=float),
                "log_scales": np.zeros((int(sequence["truth"].shape[0]), 6), dtype=float),
            }
            oracle_result = run_gm_residual_lstm_ekf(
                sequence,
                models[0][1],
                config,
                use_lstm_r_head=False,
                use_r_cm_correction=False,
                use_nis_feedback=False,
                propagate_residual_uncertainty=False,
                precomputed_shadow_features=fixed_result["features"],
                precomputed_model_outputs=oracle_outputs,
            )
            metric_rows.append(
                _metric_row(
                    scenario,
                    "GM-9D EKF + oracle residual",
                    test_seed,
                    None,
                    sequence,
                    oracle_result,
                    config,
                )
            )

            for training_seed, model in models:
                lstm_outputs = precompute_lstm_outputs(model, fixed_result["features"])
                learned_result = run_gm_residual_lstm_ekf(
                    sequence,
                    model,
                    config,
                    precomputed_shadow_features=fixed_result["features"],
                    precomputed_model_outputs=lstm_outputs,
                )
                metric_rows.append(
                    _metric_row(
                        scenario,
                        "GM-Res-LSTM-AEKF",
                        test_seed,
                        training_seed,
                        sequence,
                        learned_result,
                        config,
                    )
                )
                diagnostic_rows.append(
                    _diagnostic_row(
                        scenario,
                        test_seed,
                        training_seed,
                        sequence,
                        learned_result,
                        lstm_outputs,
                        config,
                    )
                )
            if index == 0 or (index + 1) % 5 == 0 or index + 1 == args.runs:
                print(f"  {index + 1}/{args.runs} complete")

    metric_fields = list(metric_rows[0])
    diagnostic_fields = list(diagnostic_rows[0])
    with (output_dir / "oracle_residual_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(metric_rows)
    with (output_dir / "residual_correlation_rows.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=diagnostic_fields)
        writer.writeheader()
        writer.writerows(diagnostic_rows)

    metric_summary = {}
    for scenario in scenarios:
        metric_summary[scenario] = {}
        scenario_rows = [row for row in metric_rows if row["scenario"] == scenario]
        for method in sorted({row["method"] for row in scenario_rows}):
            method_rows = [row for row in scenario_rows if row["method"] == method]
            if method == "GM-Res-LSTM-AEKF":
                per_seed_means = []
                for training_seed in sorted({row["training_seed"] for row in method_rows}):
                    seed_rows = [row for row in method_rows if row["training_seed"] == training_seed]
                    per_seed_means.append(
                        {
                            metric: float(np.mean([row[metric] for row in seed_rows]))
                            for metric in (
                                "position_rss_rmse_m",
                                "velocity_rss_rmse_mps",
                                "normalized_nis",
                                "normalized_nees",
                            )
                        }
                    )
                metric_summary[scenario][method] = {
                    metric: _summarize([row[metric] for row in per_seed_means])
                    for metric in per_seed_means[0]
                }
            else:
                metric_summary[scenario][method] = {
                    metric: _summarize([row[metric] for row in method_rows])
                    for metric in (
                        "position_rss_rmse_m",
                        "velocity_rss_rmse_mps",
                        "normalized_nis",
                        "normalized_nees",
                    )
                }

    oracle_gap_summary = {}
    for scenario in scenarios:
        scenario_summary = metric_summary[scenario]
        gm = scenario_summary["GM-9D EKF"]
        oracle = scenario_summary["GM-9D EKF + oracle residual"]
        learned = scenario_summary["GM-Res-LSTM-AEKF"]
        oracle_gap_summary[scenario] = {}
        for metric in ("position_rss_rmse_m", "velocity_rss_rmse_mps"):
            gm_mean = float(gm[metric]["mean"])
            oracle_mean = float(oracle[metric]["mean"])
            learned_mean = float(learned[metric]["mean"])
            oracle_improvement = gm_mean - oracle_mean
            learned_improvement = gm_mean - learned_mean
            oracle_gap_summary[scenario][metric] = {
                "gm_9d_mean": gm_mean,
                "oracle_residual_mean": oracle_mean,
                "gm_res_lstm_mean": learned_mean,
                "oracle_improvement": oracle_improvement,
                "learned_improvement": learned_improvement,
                "learned_fraction_of_oracle_gap": (
                    learned_improvement / oracle_improvement
                    if abs(oracle_improvement) > 1e-30
                    else float("nan")
                ),
            }

    gap_fields = [
        "scenario",
        "metric",
        "gm_9d_mean",
        "oracle_residual_mean",
        "gm_res_lstm_mean",
        "oracle_improvement",
        "learned_improvement",
        "learned_fraction_of_oracle_gap",
    ]
    with (output_dir / "oracle_gap_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=gap_fields)
        writer.writeheader()
        for scenario, metrics in oracle_gap_summary.items():
            for metric, values in metrics.items():
                writer.writerow({"scenario": scenario, "metric": metric, **values})

    diagnostic_summary = {}
    for scenario in scenarios:
        rows = [row for row in diagnostic_rows if row["scenario"] == scenario]
        diagnostic_summary[scenario] = {
            key: _summarize([row[key] for row in rows])
            for key in (
                "total_acceleration_corr_mean",
                "same_time_residual_corr_mean",
                "interval_injected_residual_corr_mean",
                "learned_to_zero_mse_ratio",
                "clip_fraction",
                "true_residual_rms_mps2",
                "learned_residual_rms_mps2",
            )
        }
        diagnostic_summary[scenario]["best_lag_steps_flat"] = {
            str(lag): int(
                sum(row["best_lag_steps_xyz"].count(lag) for row in rows)
            )
            for lag in range(-5, 6)
        }

    summary = {
        "mode": args.mode,
        "runs": args.runs,
        "scenarios": scenarios,
        "gm_tau_s": config.gm_tau_s,
        "gm_drive_psd_m2_s5": config.gm_drive_psd,
        "metric_summary": metric_summary,
        "oracle_gap_summary": oracle_gap_summary,
        "diagnostic_summary": diagnostic_summary,
        "output_files": {
            "metrics_csv": str((output_dir / "oracle_residual_metrics.csv").resolve()),
            "diagnostics_csv": str((output_dir / "residual_correlation_rows.csv").resolve()),
            "oracle_gap_csv": str((output_dir / "oracle_gap_summary.csv").resolve()),
        },
    }
    (output_dir / "oracle_residual_diagnostics.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
