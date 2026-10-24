from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp

from .build_dataset import build_gm_residual_features
from .compute_metrics import compute_segmented_metrics
from .dual_head_lstm import DualHeadLSTM
from .experiment_matrix import CORE_METRICS, holm_adjust, summarize_metric_values
from .run_adaptive_ekf import precompute_lstm_outputs, run_adaptive_ekf
from .run_classical_aekf import (
    run_coupled_nis_aekf,
    run_innovation_cm_aekf,
    run_sage_husa_fading_aekf,
)
from .run_fixed_ekf import run_fixed_ekf
from .run_adaptive_gm_ekf import run_adaptive_gm_ekf
from .run_gm_ekf import run_gm_9d_ekf
from .run_gm_residual_lstm_ekf import run_gm_residual_lstm_ekf
from .run_known_scale_ekf import run_known_scale_ekf
from .save_model import load_model, save_model
from .simulate_sequence import simulate_sequence
from .train_model import train_model


DEFAULT_TRAINING_SEEDS = (20260719, 20260791, 20260863, 20260935, 20261007)
PRIMARY_METRICS = (
    "position_rss_rmse_m",
    "velocity_rss_rmse_mps",
    "normalized_nis",
    "normalized_nees",
)
LEARNED_VARIANTS = {
    "PC-DH-LSTM-AEKF": {},
    "GM-9D EKF + LSTM residual": {
        "runner": "gm_residual",
        "use_lstm_r_head": False,
        "use_nis_feedback": False,
        "use_r_cm_correction": True,
    },
    "GM-9D EKF + LSTM residual (no residual head)": {
        "runner": "gm_residual",
        "use_residual_head": False,
        "use_lstm_r_head": False,
        "use_nis_feedback": False,
        "use_r_cm_correction": True,
    },
    "GM-9D EKF + LSTM residual (with LSTM R head)": {
        "runner": "gm_residual",
        "use_lstm_r_head": True,
        "use_nis_feedback": True,
        "use_r_cm_correction": True,
    },
    "Shared NIS Q/R in full model": {"use_qr_separation": False},
    "No acceleration head": {"use_acceleration_head": False},
    "No R fast channel": {
        "use_r_head": False,
        "use_r_fast_channel": False,
        "use_r_cm_correction": False,
        "use_nis_feedback": False,
    },
    "No Q slow constraint": {
        "use_q_slow_constraint": False,
        "use_q_cm_correction": False,
    },
    "Deterministic acceleration mean only": {
        "propagate_acceleration_uncertainty": False,
    },
}
ADAPTIVE_GM_VARIANTS = {
    "Adaptive GM-AEKF": {},
    "Adaptive GM-AEKF (fixed tau)": {"adapt_tau": False},
    "Adaptive GM-AEKF (fixed drive)": {"adapt_drive": False},
    "Adaptive GM-AEKF (no innovation corr)": {
        "use_innovation_correlation": False,
    },
    "Adaptive GM-AEKF (no accel-diff trigger)": {
        "use_acceleration_difference": False,
    },
    "Adaptive GM-AEKF (no R-CM)": {"adapt_r": False},
}
FACTOR_SPECS = {
    "amplitude_scale": {
        "levels": (0.5, 1.0, 1.5, 2.0, 3.0),
        "variant_factor": "amplitude",
        "rms_match": False,
        "unit": "x",
    },
    "frequency_scale": {
        "levels": (0.5, 1.0, 1.5, 2.5, 4.0),
        "variant_factor": "frequency",
        "rms_match": True,
        "unit": "x",
    },
    "transition_duration_s": {
        "levels": (0.0, 1.0, 5.0, 15.0, 30.0),
        "variant_factor": "transition_duration",
        "rms_match": True,
        "unit": "s",
    },
    "measurement_interval_s": {
        "levels": (0.05, 0.1, 0.2, 0.5),
        "variant_factor": None,
        "rms_match": False,
        "unit": "s",
    },
}

_WORKER_CONTEXT: dict[str, object] = {}


def _run_learned_variant(
    method: str,
    options: dict[str, object],
    sequence,
    model,
    residual_model,
    config,
    fixed_features: np.ndarray,
    gm_residual_features: np.ndarray,
    precomputed_outputs: dict[str, np.ndarray],
    precomputed_residual_outputs: dict[str, np.ndarray],
):
    runner = str(options.get("runner", "pcdh"))
    call_options = {key: value for key, value in options.items() if key != "runner"}
    if runner == "gm_residual":
        return run_gm_residual_lstm_ekf(
            sequence,
            residual_model,
            config,
            precomputed_shadow_features=gm_residual_features,
            precomputed_model_outputs=precomputed_residual_outputs,
            **call_options,
        )
    if runner == "pcdh":
        return run_adaptive_ekf(
            sequence,
            model,
            config,
            precomputed_shadow_features=fixed_features,
            precomputed_model_outputs=precomputed_outputs,
            **call_options,
        )
    raise ValueError(f"Unknown learned variant runner for {method}: {runner}")


def _initialize_experiment_worker(
    config,
    models: list[DualHeadLSTM],
    residual_models: list[DualHeadLSTM],
    training_seeds: tuple[int, ...],
    learned_variants: dict[str, dict[str, object]],
    adaptive_gm_variants: dict[str, dict[str, object]] | None = None,
) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = {
        "config": config,
        "models": models,
        "residual_models": residual_models,
        "training_seeds": training_seeds,
        "learned_variants": learned_variants,
        "adaptive_gm_variants": adaptive_gm_variants or {},
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_model_ensemble(
    config,
    features: np.ndarray,
    labels: np.ndarray,
    training_seeds: tuple[int, ...],
    output_dir: Path,
    *,
    show_progress: bool = False,
    resume_existing: bool = False,
    model_kind: str = "total_acceleration",
) -> tuple[list[DualHeadLSTM], list[dict[str, object]]]:
    """Train independent initializations on one fixed data split.

    Completed weight archives can be reused after an interrupted long run.  A
    sidecar record is written after every newly trained model so subsequent
    resumes retain the full training history.
    """

    if len(training_seeds) < 5:
        raise ValueError("Major-revision reporting requires at least five training seeds")
    output_dir.mkdir(parents=True, exist_ok=True)
    if model_kind not in {
        "total_acceleration",
        "gm_residual",
        "gm_total_acceleration",
    }:
        raise ValueError(f"Unknown model_kind: {model_kind}")
    prefix = "dual_head_lstm" if model_kind == "total_acceleration" else "gm_residual_lstm"
    models = []
    records = []
    for index, training_seed in enumerate(training_seeds, start=1):
        model_path = output_dir / f"{prefix}_trainseed_{training_seed}.npz"
        record_path = output_dir / f"{prefix}_trainseed_{training_seed}.json"
        if resume_existing and model_path.exists():
            if show_progress:
                print(
                    f"  Reusing training instance {index}/{len(training_seeds)} "
                    f"(seed={training_seed})"
                )
            model = load_model(model_path)
            if model.output_mode != model_kind or model.input_size != features.shape[-1]:
                raise ValueError(
                    f"Stored model specification mismatch: {model_path}"
                )
            if record_path.exists():
                record = json.loads(record_path.read_text(encoding="utf-8"))
            else:
                split_rng = np.random.default_rng(config.seed)
                split_order = split_rng.permutation(features.shape[0])
                validation_count = int(
                    round(features.shape[0] * config.validation_fraction)
                )
                validation_count = min(
                    max(validation_count, 1), features.shape[0] - 1
                )
                validation_indices = np.sort(split_order[:validation_count])
                training_indices = np.sort(split_order[validation_count:])
                validation_loss = model.evaluate_loss(
                    features[validation_indices], labels[validation_indices]
                )["total"]
                history = {
                    "training_loss": [],
                    "validation_loss": [],
                    "best_epoch": None,
                    "best_validation_loss": float(validation_loss),
                    "training_indices": training_indices.tolist(),
                    "validation_indices": validation_indices.tolist(),
                    "training_sequences": int(training_indices.size),
                    "validation_sequences": int(validation_indices.size),
                    "training_seed": int(training_seed),
                    "split_seed": int(config.seed),
                    "acceleration_error_covariance_m2_s4": (
                        model.acceleration_error_covariance.tolist()
                    ),
                    "acceleration_error_covariance_source": (
                        "validation residual second moment stored in weight archive"
                    ),
                    "training_curve_available": False,
                    "recovery_note": (
                        "Complete best-weight archive recovered after runner-pipe "
                        "interruption; epoch-wise losses were not persisted."
                    ),
                }
                record = {
                    "training_seed": int(training_seed),
                    "model_path": str(model_path.resolve()),
                    "sha256": sha256_file(model_path),
                    "history": history,
                }
                record_path.write_text(
                    json.dumps(record, indent=2, ensure_ascii=True),
                    encoding="utf-8",
                )
            if int(record["training_seed"]) != training_seed:
                raise ValueError(f"Sidecar seed mismatch: {record_path}")
            if record["sha256"] != sha256_file(model_path):
                raise ValueError(f"Sidecar hash mismatch: {model_path}")
            models.append(model)
            records.append(record)
            continue

        if show_progress:
            print(
                f"  Training instance {index}/{len(training_seeds)} "
                f"(seed={training_seed})"
            )
        residual_model = model_kind in {"gm_residual", "gm_total_acceleration"}
        model = DualHeadLSTM(
            config.residual_hidden_size if residual_model else config.hidden_size,
            config.a_max,
            config.log_scale_max,
            seed=training_seed,
            input_size=features.shape[-1],
            output_mode=model_kind,
            loss_weights=(
                {
                    "acceleration": config.residual_loss_acceleration_weight,
                    "covariance": config.residual_loss_covariance_weight,
                    "smoothness": config.residual_loss_smoothness_weight,
                    "acceleration_smoothness": (
                        config.residual_loss_acceleration_smoothness_weight
                    ),
                }
                if residual_model
                else None
            ),
        )
        training_config = copy.deepcopy(config)
        if residual_model:
            training_config.epochs = config.residual_epochs
            training_config.learning_rate = config.residual_learning_rate
            training_config.training_window_steps = (
                config.residual_training_window_steps
            )
            training_config.training_windows_per_sequence = (
                config.residual_training_windows_per_sequence
            )
            training_config.early_stopping_patience = (
                config.residual_early_stopping_patience
            )
        history = train_model(
            model,
            features,
            labels,
            training_config,
            training_seed=training_seed,
            show_progress=show_progress,
        )
        save_model(model, model_path)
        record = {
            "training_seed": int(training_seed),
            "model_kind": model_kind,
            "input_size": int(model.input_size),
            "model_path": str(model_path.resolve()),
            "sha256": sha256_file(model_path),
            "history": history,
        }
        record_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        models.append(model)
        records.append(record)
    return models, records


def calibrate_gm_residual_injection(
    model: DualHeadLSTM,
    validation_sequences: list[dict[str, object]],
    config,
    *,
    gain_candidates: tuple[float, ...] = (
        0.0,
        0.10,
        0.20,
        0.35,
        0.50,
        0.70,
        1.0,
    ),
) -> dict[str, object]:
    """Freeze a validation-only residual-injection multiplier.

    Candidate zero is intentionally included, so an unreliable learner falls
    back exactly to GM-9D instead of degrading the navigation solution.
    """

    if not validation_sequences:
        raise ValueError("Residual calibration requires validation trajectories")
    base_gain = model.residual_injection_gain.copy()
    baseline_metrics = []
    cached = []
    for sequence in validation_sequences:
        gm_result = run_gm_9d_ekf(sequence, config, adapt_r=True)
        features = build_gm_residual_features(gm_result, config)
        outputs = precompute_lstm_outputs(model, features)
        baseline_result = run_gm_residual_lstm_ekf(
            sequence,
            model,
            config,
            use_residual_head=False,
            use_lstm_r_head=False,
            use_r_cm_correction=True,
            use_nis_feedback=False,
            propagate_residual_uncertainty=False,
            precomputed_shadow_features=features,
            precomputed_model_outputs=outputs,
        )
        metrics = compute_segmented_metrics(sequence, baseline_result, config)[
            "post_burn_in"
        ]
        baseline_metrics.append(metrics)
        cached.append((sequence, features, outputs))

    candidate_rows = []
    for multiplier in gain_candidates:
        model.residual_injection_gain = base_gain * float(multiplier)
        relative_scores = []
        for (sequence, features, outputs), baseline_metric in zip(
            cached, baseline_metrics, strict=True
        ):
            result = run_gm_residual_lstm_ekf(
                sequence,
                model,
                config,
                use_lstm_r_head=False,
                use_r_cm_correction=True,
                use_nis_feedback=False,
                propagate_residual_uncertainty=False,
                precomputed_shadow_features=features,
                precomputed_model_outputs=outputs,
            )
            metric = compute_segmented_metrics(sequence, result, config)[
                "post_burn_in"
            ]
            relative_scores.append(
                0.5
                * (
                    metric["position_rss_rmse_m"]
                    / max(baseline_metric["position_rss_rmse_m"], 1e-12)
                    + metric["velocity_rss_rmse_mps"]
                    / max(baseline_metric["velocity_rss_rmse_mps"], 1e-12)
                )
            )
        candidate_rows.append(
            {
                "multiplier": float(multiplier),
                "mean_relative_accuracy_score": float(np.mean(relative_scores)),
            }
        )
    selected = min(
        candidate_rows,
        key=lambda row: (row["mean_relative_accuracy_score"], row["multiplier"]),
    )
    model.residual_injection_gain = base_gain * selected["multiplier"]
    return {
        "selection_data": "LSTM validation trajectories only",
        "selection_metric": (
            "mean post-burn-in 0.5 * (position RMSE / matched no-residual "
            "position RMSE + velocity RMSE / matched no-residual velocity RMSE)"
        ),
        "base_axis_gain": base_gain.tolist(),
        "selected_multiplier": selected["multiplier"],
        "selected_axis_gain": model.residual_injection_gain.tolist(),
        "candidate_grid": candidate_rows,
        "test_set_used_for_selection": False,
    }


def _metric_row(
    method: str,
    test_seed: int,
    segmented: dict[str, dict],
    *,
    training_seed: int | None,
) -> dict[str, object]:
    return {
        "method": method,
        "training_seed": training_seed,
        "test_seed": int(test_seed),
        "metrics": {
            phase: {metric: values[metric] for metric in CORE_METRICS}
            for phase, values in segmented.items()
        },
    }


def _summarize_crossed_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    training_seeds = sorted(
        {int(row["training_seed"]) for row in rows if row["training_seed"] is not None}
    )
    if not training_seeds:
        return {
            phase: {
                metric: summarize_metric_values(
                    [float(row["metrics"][phase][metric]) for row in rows]
                )
                for metric in CORE_METRICS
            }
            for phase in rows[0]["metrics"]
        }

    per_training_seed = {}
    per_training_seed_statistics = {}
    for training_seed in training_seeds:
        seed_rows = [row for row in rows if row["training_seed"] == training_seed]
        per_training_seed[str(training_seed)] = {
            phase: {
                metric: float(
                    np.mean([row["metrics"][phase][metric] for row in seed_rows])
                )
                for metric in CORE_METRICS
            }
            for phase in seed_rows[0]["metrics"]
        }
        per_training_seed_statistics[str(training_seed)] = {
            phase: {
                metric: summarize_metric_values(
                    [float(row["metrics"][phase][metric]) for row in seed_rows]
                )
                for metric in CORE_METRICS
            }
            for phase in seed_rows[0]["metrics"]
        }
    across_training_seeds = {
        phase: {
            metric: summarize_metric_values(
                [per_training_seed[str(seed)][phase][metric] for seed in training_seeds]
            )
            for metric in CORE_METRICS
        }
        for phase in rows[0]["metrics"]
    }
    return {
        "inference_unit": "one test-seed mean per independent training seed",
        "training_seed_count": len(training_seeds),
        "test_seed_count_per_training_seed": len(rows) // len(training_seeds),
        "per_training_seed_test_mean": per_training_seed,
        "per_training_seed_test_statistics": per_training_seed_statistics,
        "pooled_descriptive_statistics": {
            "warning": (
                "The n=training_seed_count*test_seed_count rows are correlated; "
                "do not use this pooled CI for training-level inference."
            ),
            "phases": {
                phase: {
                    metric: summarize_metric_values(
                        [float(row["metrics"][phase][metric]) for row in rows]
                    )
                    for metric in CORE_METRICS
                }
                for phase in rows[0]["metrics"]
            },
        },
        "across_training_seeds": across_training_seeds,
    }


def _holm_ablation_tests(
    rows_by_method: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    reference = rows_by_method["PC-DH-LSTM-AEKF"]
    reference_lookup = {
        (row["training_seed"], row["test_seed"]): row for row in reference
    }
    tests = []
    p_values = []
    for method, rows in rows_by_method.items():
        if method == "PC-DH-LSTM-AEKF":
            continue
        training_seeds = sorted({int(row["training_seed"]) for row in rows})
        for metric in PRIMARY_METRICS:
            seed_mean_differences = []
            for training_seed in training_seeds:
                differences = [
                    row["metrics"]["post_burn_in"][metric]
                    - reference_lookup[(training_seed, row["test_seed"])]["metrics"]
                    ["post_burn_in"][metric]
                    for row in rows
                    if row["training_seed"] == training_seed
                ]
                seed_mean_differences.append(float(np.mean(differences)))
            p_value = float(ttest_1samp(seed_mean_differences, 0.0).pvalue)
            if not np.isfinite(p_value):
                p_value = 1.0
            tests.append(
                {
                    "method": method,
                    "metric": metric,
                    "phase": "post_burn_in",
                    "training_seed_mean_differences": seed_mean_differences,
                    "difference_summary": summarize_metric_values(seed_mean_differences),
                    "raw_p_value": p_value,
                }
            )
            p_values.append(p_value)
    for test, adjusted in zip(tests, holm_adjust(p_values), strict=True):
        test["holm_adjusted_p_value"] = adjusted
        test["reject_at_0_05"] = bool(adjusted < 0.05)
    return {
        "inference_unit": "training-seed mean paired difference over common test seeds",
        "correction": "Holm family-wise error correction",
        "family_size": len(tests),
        "alpha": 0.05,
        "tests": tests,
    }


def _holm_adaptive_gm_ablation_tests(
    rows_by_method: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    reference = rows_by_method["Adaptive GM-AEKF"]
    reference_lookup = {int(row["test_seed"]): row for row in reference}
    tests = []
    p_values = []
    for method, rows in rows_by_method.items():
        if method == "Adaptive GM-AEKF":
            continue
        for metric in PRIMARY_METRICS:
            differences = [
                row["metrics"]["post_burn_in"][metric]
                - reference_lookup[int(row["test_seed"])]["metrics"]["post_burn_in"][
                    metric
                ]
                for row in rows
            ]
            p_value = float(ttest_1samp(differences, 0.0).pvalue)
            if not np.isfinite(p_value):
                p_value = 1.0
            tests.append(
                {
                    "method": method,
                    "metric": metric,
                    "phase": "post_burn_in",
                    "test_seed_paired_differences": differences,
                    "difference_summary": summarize_metric_values(differences),
                    "raw_p_value": p_value,
                }
            )
            p_values.append(p_value)
    for test, adjusted in zip(tests, holm_adjust(p_values), strict=True):
        test["holm_adjusted_p_value"] = adjusted
        test["reject_at_0_05"] = bool(adjusted < 0.05)
    return {
        "inference_unit": "test-seed paired difference over common trajectories",
        "correction": "Holm family-wise error correction",
        "family_size": len(tests),
        "alpha": 0.05,
        "reference_method": "Adaptive GM-AEKF",
        "difference_definition": "method minus Adaptive GM-AEKF",
        "tests": tests,
    }


def _evaluate_crossed_seed(
    config,
    models: list[DualHeadLSTM],
    residual_models: list[DualHeadLSTM],
    training_seeds: tuple[int, ...],
    learned_variants: dict[str, dict[str, object]],
    adaptive_gm_variants: dict[str, dict[str, object]],
    test_seed: int,
    scenario: str,
) -> dict[str, list[dict[str, object]]]:
    sequence = simulate_sequence(config, test_seed, scenario)
    fixed_result = run_fixed_ekf(sequence, config)
    gm_result = run_gm_9d_ekf(sequence, config)
    gm_r_cm_result = run_gm_9d_ekf(sequence, config, adapt_r=True)
    gm_residual_features = build_gm_residual_features(gm_r_cm_result, config)
    baseline_results = {
        "Fixed EKF": fixed_result,
        "Innovation-CM AEKF": run_innovation_cm_aekf(sequence, config),
        "Sage-Husa/Fading AEKF": run_sage_husa_fading_aekf(sequence, config),
        "Coupled-NIS Q/R AEKF": run_coupled_nis_aekf(sequence, config),
        "GM-9D EKF": gm_result,
        "GM-9D EKF + R-CM": gm_r_cm_result,
        "Known-scale EKF": run_known_scale_ekf(sequence, config),
    }
    adaptive_results = {
        method: run_adaptive_gm_ekf(sequence, config, **options)
        for method, options in adaptive_gm_variants.items()
    }
    rows = {
        method: []
        for method in (*baseline_results, *adaptive_results, *learned_variants)
    }
    for method, result in baseline_results.items():
        rows[method].append(
            _metric_row(
                method,
                test_seed,
                compute_segmented_metrics(sequence, result, config),
                training_seed=None,
            )
        )
    for method, result in adaptive_results.items():
        rows[method].append(
            _metric_row(
                method,
                test_seed,
                compute_segmented_metrics(sequence, result, config),
                training_seed=None,
            )
        )
    for training_seed, model, residual_model in zip(
        training_seeds, models, residual_models, strict=True
    ):
        precomputed_outputs = precompute_lstm_outputs(model, fixed_result["features"])
        precomputed_residual_outputs = precompute_lstm_outputs(
            residual_model, gm_residual_features
        )
        for method, options in learned_variants.items():
            result = _run_learned_variant(
                method,
                options,
                sequence,
                model,
                residual_model,
                config,
                fixed_result["features"],
                gm_residual_features,
                precomputed_outputs,
                precomputed_residual_outputs,
            )
            rows[method].append(
                _metric_row(
                    method,
                    test_seed,
                    compute_segmented_metrics(sequence, result, config),
                    training_seed=training_seed,
                )
            )
    return rows


def _crossed_seed_worker(payload: tuple[int, str]):
    test_seed, scenario = payload
    return _evaluate_crossed_seed(
        _WORKER_CONTEXT["config"],
        _WORKER_CONTEXT["models"],
        _WORKER_CONTEXT["residual_models"],
        _WORKER_CONTEXT["training_seeds"],
        _WORKER_CONTEXT["learned_variants"],
        _WORKER_CONTEXT["adaptive_gm_variants"],
        test_seed,
        scenario,
    )


def run_crossed_scenario_experiment(
    config,
    models: list[DualHeadLSTM],
    residual_models: list[DualHeadLSTM],
    training_seeds: tuple[int, ...],
    *,
    scenario: str,
    runs: int,
    seed_offset: int,
    include_ablation: bool,
    workers: int = 1,
    show_progress: bool = False,
) -> dict[str, object]:
    """Run a crossed training-seed by test-seed experiment without pseudoreplication."""

    if (
        len(models) != len(training_seeds)
        or len(residual_models) != len(training_seeds)
        or len(models) < 5
    ):
        raise ValueError("Exactly one model is required for each of at least five seeds")
    if runs < 30:
        raise ValueError("At least 30 common test seeds are required")

    learned_variants = LEARNED_VARIANTS if include_ablation else {
        "PC-DH-LSTM-AEKF": LEARNED_VARIANTS["PC-DH-LSTM-AEKF"],
        "GM-9D EKF + LSTM residual": LEARNED_VARIANTS[
            "GM-9D EKF + LSTM residual"
        ],
    }
    adaptive_gm_variants = (
        ADAPTIVE_GM_VARIANTS
        if include_ablation
        else {"Adaptive GM-AEKF": ADAPTIVE_GM_VARIANTS["Adaptive GM-AEKF"]}
    )
    rows_by_method: dict[str, list[dict[str, object]]] = {
        "Fixed EKF": [],
        "Innovation-CM AEKF": [],
        "Sage-Husa/Fading AEKF": [],
        "Coupled-NIS Q/R AEKF": [],
        "GM-9D EKF": [],
        "GM-9D EKF + R-CM": [],
        **{name: [] for name in adaptive_gm_variants},
        "Known-scale EKF": [],
        **{name: [] for name in learned_variants},
    }
    test_seeds = [int(config.seed + seed_offset + index) for index in range(runs)]
    payloads = [(test_seed, scenario) for test_seed in test_seeds]
    if workers > 1:
        executor_context = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_experiment_worker,
            initargs=(
                config,
                models,
                residual_models,
                training_seeds,
                learned_variants,
                adaptive_gm_variants,
            ),
        )
        result_iterator = executor_context.map(_crossed_seed_worker, payloads, chunksize=1)
    else:
        executor_context = None
        result_iterator = (
            _evaluate_crossed_seed(
                config,
                models,
                residual_models,
                training_seeds,
                learned_variants,
                adaptive_gm_variants,
                test_seed,
                scenario,
            )
            for test_seed in test_seeds
        )
    try:
        for run_index, seed_rows in enumerate(result_iterator, start=1):
            for method, rows in seed_rows.items():
                rows_by_method[method].extend(rows)
            if show_progress and (
                run_index == 1 or run_index % 5 == 0 or run_index == runs
            ):
                print(f"  Crossed {scenario}: {run_index}/{runs} test seeds complete")
    finally:
        if executor_context is not None:
            executor_context.shutdown(wait=True, cancel_futures=False)

    summary = {
        method: _summarize_crossed_rows(rows) for method, rows in rows_by_method.items()
    }
    result = {
        "scenario": scenario,
        "training_seeds": list(training_seeds),
        "test_seeds": test_seeds,
        "test_seed_count": runs,
        "design": "crossed fixed-data 5-training-seed by 30-test-seed experiment",
        "rows_by_method": rows_by_method,
        "summary": summary,
    }
    if include_ablation:
        learned_rows = {method: rows_by_method[method] for method in learned_variants}
        result["confirmatory_ablation_tests"] = _holm_ablation_tests(learned_rows)
        adaptive_rows = {
            method: rows_by_method[method] for method in adaptive_gm_variants
        }
        result["adaptive_gm_ablation_tests"] = _holm_adaptive_gm_ablation_tests(
            adaptive_rows
        )
    return result


def _measurement_interval_config(config, dt: float):
    adjusted = copy.deepcopy(config)
    base_dt = float(config.dt)
    duration = (config.steps - 1) * base_dt
    adjusted.dt = float(dt)
    adjusted.steps = int(round(duration / dt)) + 1
    ratio = dt / base_dt
    for field in ("r_cm_window", "q_cm_window", "baseline_cm_window"):
        setattr(adjusted, field, max(4, int(round(getattr(config, field) / ratio))))
    for field in ("ema_beta", "r_ema_beta", "q_lstm_ema_beta", "baseline_coupled_ema_beta"):
        setattr(adjusted, field, float(getattr(config, field) ** ratio))
    for field in ("r_max_log_step", "q_max_log_step", "baseline_coupled_max_log_step"):
        setattr(adjusted, field, float(getattr(config, field) * ratio))
    for field in ("r_update_alpha", "q_lstm_update_alpha", "baseline_cm_alpha"):
        base_alpha = float(getattr(config, field))
        setattr(adjusted, field, 1.0 - (1.0 - base_alpha) ** ratio)
    return adjusted


def _factor_sequence(config, test_seed: int, factor: str, level: float):
    specification = FACTOR_SPECS[factor]
    if factor == "measurement_interval_s":
        level_config = _measurement_interval_config(config, level)
        return level_config, simulate_sequence(
            level_config,
            test_seed,
            "nonlinear_structured",
        )
    variant = {
        "factor": specification["variant_factor"],
        "level": level,
        "rms_match": specification["rms_match"],
    }
    return config, simulate_sequence(
        config,
        test_seed,
        "nonlinear_structured",
        acceleration_variant=variant,
    )


def _evaluate_factor_task(
    config,
    models: list[DualHeadLSTM],
    residual_models: list[DualHeadLSTM],
    training_seeds: tuple[int, ...],
    factor: str,
    level: float,
    test_seed: int,
) -> list[dict[str, object]]:
    level_config, sequence = _factor_sequence(config, test_seed, factor, level)
    fixed_result = run_fixed_ekf(sequence, level_config)
    gm_result = run_gm_9d_ekf(sequence, level_config)
    gm_r_cm_result = run_gm_9d_ekf(sequence, level_config, adapt_r=True)
    adaptive_gm_result = run_adaptive_gm_ekf(sequence, level_config)
    gm_residual_features = build_gm_residual_features(
        gm_r_cm_result, level_config
    )
    baseline_results = {
        "Fixed EKF": fixed_result,
        "GM-9D EKF": gm_result,
        "Adaptive GM-AEKF": adaptive_gm_result,
    }
    rows = []
    for method, result in baseline_results.items():
        segmented = compute_segmented_metrics(sequence, result, level_config)
        rows.append(
            {
                "factor": factor,
                "level": float(level),
                **_metric_row(
                    method,
                    test_seed,
                    segmented,
                    training_seed=None,
                ),
            }
        )
    for training_seed, model, residual_model in zip(
        training_seeds, models, residual_models, strict=True
    ):
        precomputed_outputs = precompute_lstm_outputs(model, fixed_result["features"])
        precomputed_residual_outputs = precompute_lstm_outputs(
            residual_model, gm_residual_features
        )
        learned_methods = {
            "PC-DH-LSTM-AEKF": LEARNED_VARIANTS["PC-DH-LSTM-AEKF"],
            "GM-9D EKF + LSTM residual": LEARNED_VARIANTS[
                "GM-9D EKF + LSTM residual"
            ],
        }
        for method, options in learned_methods.items():
            result = _run_learned_variant(
                method,
                options,
                sequence,
                model,
                residual_model,
                level_config,
                fixed_result["features"],
                gm_residual_features,
                precomputed_outputs,
                precomputed_residual_outputs,
            )
            segmented = compute_segmented_metrics(sequence, result, level_config)
            rows.append(
                {
                    "factor": factor,
                    "level": float(level),
                    **_metric_row(
                        method,
                        test_seed,
                        segmented,
                        training_seed=training_seed,
                    ),
                }
            )
    return rows


def _factor_task_worker(payload: tuple[str, float, int]):
    factor, level, test_seed = payload
    return _evaluate_factor_task(
        _WORKER_CONTEXT["config"],
        _WORKER_CONTEXT["models"],
        _WORKER_CONTEXT["residual_models"],
        _WORKER_CONTEXT["training_seeds"],
        factor,
        level,
        test_seed,
    )


def _factor_task_key(payload: tuple[str, float, int]) -> str:
    factor, level, test_seed = payload
    return f"{factor}|{level:.17g}|{test_seed}"


FACTOR_SCAN_METHODS = {
    "Fixed EKF",
    "GM-9D EKF",
    "Adaptive GM-AEKF",
    "PC-DH-LSTM-AEKF",
    "GM-9D EKF + LSTM residual",
}


def _load_factor_checkpoint(
    path: Path,
    *,
    required_methods: set[str] | None = None,
) -> dict[str, list[dict[str, object]]]:
    completed = {}
    if not path.exists():
        return completed
    lines = path.read_text(encoding="utf-8").splitlines()
    valid_lines = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            if line_number == len(lines):
                path.write_text(
                    "\n".join(valid_lines) + ("\n" if valid_lines else ""),
                    encoding="utf-8",
                )
                break
            raise ValueError(
                f"Invalid factor checkpoint JSON at {path}:{line_number}"
            ) from error
        valid_lines.append(line)
        payload = (
            str(record["factor"]),
            float(record["level"]),
            int(record["test_seed"]),
        )
        key = _factor_task_key(payload)
        if key != record["task_key"]:
            raise ValueError(f"Factor checkpoint key mismatch at {path}:{line_number}")
        if required_methods is not None:
            recorded_methods = {row["method"] for row in record["rows"]}
            if not required_methods.issubset(recorded_methods):
                continue
        completed[key] = record["rows"]
    return completed


def run_factor_scan(
    config,
    models: list[DualHeadLSTM],
    residual_models: list[DualHeadLSTM],
    training_seeds: tuple[int, ...],
    *,
    runs: int,
    seed_offset: int = 3_000_000,
    workers: int = 1,
    show_progress: bool = False,
    checkpoint_path: Path | None = None,
) -> dict[str, object]:
    """Scan four factors using common test seeds and five independent models."""

    if (
        len(models) != len(training_seeds)
        or len(residual_models) != len(training_seeds)
        or len(models) < 5
    ):
        raise ValueError("Factor scans require at least five independent training seeds")
    if runs < 30:
        raise ValueError("Factor scans require at least 30 test seeds")
    test_seeds = [int(config.seed + seed_offset + index) for index in range(runs)]
    rows = []
    payloads = [
        (factor, float(level), test_seed)
        for factor, specification in FACTOR_SPECS.items()
        for level in specification["levels"]
        for test_seed in test_seeds
    ]
    completed_tasks = (
        _load_factor_checkpoint(
            checkpoint_path,
            required_methods=FACTOR_SCAN_METHODS,
        )
        if checkpoint_path is not None
        else {}
    )
    rows = [
        row
        for payload in payloads
        for row in completed_tasks.get(_factor_task_key(payload), [])
    ]
    pending_payloads = [
        payload
        for payload in payloads
        if _factor_task_key(payload) not in completed_tasks
    ]
    if show_progress and completed_tasks:
        print(
            f"  Factor scan checkpoint: {len(completed_tasks)}/{len(payloads)} "
            "tasks restored"
        )
    if workers > 1 and pending_payloads:
        executor_context = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_experiment_worker,
            initargs=(config, models, residual_models, training_seeds, {}, {}),
        )
        result_iterator = executor_context.map(
            _factor_task_worker, pending_payloads, chunksize=1
        )
    else:
        executor_context = None
        result_iterator = (
            _evaluate_factor_task(
                config,
                models,
                residual_models,
                training_seeds,
                factor,
                level,
                test_seed,
            )
            for factor, level, test_seed in pending_payloads
        )
    checkpoint_handle = None
    if checkpoint_path is not None and pending_payloads:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_handle = checkpoint_path.open("a", encoding="utf-8")
    try:
        for task_index, task_rows in enumerate(result_iterator, start=1):
            payload = pending_payloads[task_index - 1]
            rows.extend(task_rows)
            if checkpoint_handle is not None:
                checkpoint_handle.write(
                    json.dumps(
                        {
                            "task_key": _factor_task_key(payload),
                            "factor": payload[0],
                            "level": payload[1],
                            "test_seed": payload[2],
                            "rows": task_rows,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                checkpoint_handle.flush()
            completed_count = len(completed_tasks) + task_index
            if show_progress and (
                task_index == 1
                or completed_count % max(1, runs) == 0
                or completed_count == len(payloads)
            ):
                factor, level, _ = payload
                print(
                    f"  Factor scan: {completed_count}/{len(payloads)} tasks complete "
                    f"(through {factor}={level})"
                )
    finally:
        if checkpoint_handle is not None:
            checkpoint_handle.close()
        if executor_context is not None:
            executor_context.shutdown(wait=True, cancel_futures=False)

    factor_rank = {name: index for index, name in enumerate(FACTOR_SPECS)}
    level_rank = {
        (factor, float(level)): index
        for factor, specification in FACTOR_SPECS.items()
        for index, level in enumerate(specification["levels"])
    }
    method_rank = {
        "Fixed EKF": 0,
        "GM-9D EKF": 1,
        "Adaptive GM-AEKF": 2,
        "PC-DH-LSTM-AEKF": 3,
        "GM-9D EKF + LSTM residual": 4,
    }
    rows.sort(
        key=lambda row: (
            factor_rank[row["factor"]],
            level_rank[(row["factor"], float(row["level"]))],
            int(row["test_seed"]),
            method_rank[row["method"]],
            -1 if row["training_seed"] is None else int(row["training_seed"]),
        )
    )

    summary = {}
    crossover = {}
    for factor, specification in FACTOR_SPECS.items():
        summary[factor] = {}
        crossover[factor] = {}
        for level in specification["levels"]:
            level_rows = [
                row for row in rows if row["factor"] == factor and row["level"] == level
            ]
            summary[factor][str(level)] = {
                method: _summarize_crossed_rows(
                    [row for row in level_rows if row["method"] == method]
                )
                for method in (
                    "Fixed EKF",
                    "GM-9D EKF",
                    "Adaptive GM-AEKF",
                    "PC-DH-LSTM-AEKF",
                    "GM-9D EKF + LSTM residual",
                )
            }
            gm_lookup = {
                row["test_seed"]: row
                for row in level_rows
                if row["method"] == "GM-9D EKF"
            }
            full_rows = [
                row for row in level_rows if row["method"] == "Adaptive GM-AEKF"
            ]
            metric_results = {}
            for metric in PRIMARY_METRICS:
                paired_differences = [
                    gm_lookup[row["test_seed"]]["metrics"]["post_burn_in"][metric]
                    - row["metrics"]["post_burn_in"][metric]
                    for row in full_rows
                ]
                statistics = summarize_metric_values(paired_differences)
                if statistics["ci_lower"] > 0.0:
                    ordering = "adaptive_gm_better"
                elif statistics["ci_upper"] < 0.0:
                    ordering = "gm_better"
                else:
                    ordering = "unresolved"
                metric_results[metric] = {
                    "difference_definition": "GM-9D minus Adaptive GM-AEKF",
                    "test_seed_paired_differences": paired_differences,
                    "summary": statistics,
                    "ordering_at_95_percent": ordering,
                }
            crossover[factor][str(level)] = metric_results

    return {
        "design": "single-factor scan with 5 training seeds and 30 common test seeds",
        "base_scenario": "nonlinear_structured learnable residual stress",
        "factor_specs": FACTOR_SPECS,
        "training_seeds": list(training_seeds),
        "test_seeds": test_seeds,
        "rows": rows,
        "summary": summary,
        "gm_minus_adaptive_crossover": crossover,
    }


def save_json(data: dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_crossed_rows_csv(data: dict[str, object], path: Path) -> None:
    fields = [
        "scenario",
        "method",
        "training_seed",
        "test_seed",
        "phase",
        *CORE_METRICS,
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method_rows in data["rows_by_method"].values():
            for row in method_rows:
                for phase, metrics in row["metrics"].items():
                    writer.writerow(
                        {
                            "scenario": data["scenario"],
                            "method": row["method"],
                            "training_seed": row["training_seed"],
                            "test_seed": row["test_seed"],
                            "phase": phase,
                            **metrics,
                        }
                    )


def save_factor_rows_csv(data: dict[str, object], path: Path) -> None:
    fields = [
        "factor",
        "level",
        "method",
        "training_seed",
        "test_seed",
        "phase",
        *CORE_METRICS,
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in data["rows"]:
            for phase, metrics in row["metrics"].items():
                writer.writerow(
                    {
                        "factor": row["factor"],
                        "level": row["level"],
                        "method": row["method"],
                        "training_seed": row["training_seed"],
                        "test_seed": row["test_seed"],
                        "phase": phase,
                        **metrics,
                    }
                )
