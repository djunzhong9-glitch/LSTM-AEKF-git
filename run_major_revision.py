import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import matplotlib
import numpy as np
import scipy

from pcdh_lstm_aekf.build_dataset import (
    build_dataset,
    build_gm_residual_dataset,
    build_validation_sequences,
)
from pcdh_lstm_aekf.major_revision_experiments import (
    DEFAULT_TRAINING_SEEDS,
    calibrate_gm_residual_injection,
    run_crossed_scenario_experiment,
    run_factor_scan,
    save_crossed_rows_csv,
    save_factor_rows_csv,
    save_json,
    train_model_ensemble,
)
from pcdh_lstm_aekf.make_config import make_config
from pcdh_lstm_aekf.plot_major_revision import (
    plot_adaptive_gm_ablation,
    plot_consistency_coverage,
    plot_factor_boundaries,
    plot_holm_ablation,
    plot_training_seed_summary,
)
from pcdh_lstm_aekf.reproducibility import _json_ready
from pcdh_lstm_aekf.run_adaptive_gm_ekf import (
    apply_adaptive_gm_parameters,
    tune_adaptive_gm_hyperparameters,
)
from pcdh_lstm_aekf.run_gm_ekf import tune_gm_hyperparameters
from pcdh_lstm_aekf.save_model import load_model, save_model


REQUIRED_CROSSED_METHODS = {
    "Fixed EKF",
    "GM-9D EKF",
    "GM-9D EKF + R-CM",
    "Adaptive GM-AEKF",
    "PC-DH-LSTM-AEKF",
    "GM-9D EKF + LSTM residual",
}
REQUIRED_FACTOR_METHODS = {
    "Fixed EKF",
    "GM-9D EKF",
    "Adaptive GM-AEKF",
    "PC-DH-LSTM-AEKF",
    "GM-9D EKF + LSTM residual",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command Major Revision experiment pipeline"
    )
    parser.add_argument("--mode", choices=("quick", "paper"), default="paper")
    parser.add_argument("--mc-runs", type=int, default=30)
    parser.add_argument(
        "--training-seeds",
        default=",".join(str(seed) for seed in DEFAULT_TRAINING_SEEDS),
    )
    parser.add_argument("--resume-models", action="store_true")
    parser.add_argument(
        "--resume-results",
        action="store_true",
        help="Reuse completed GM/ID/factor JSON stages after interruption",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, (os.cpu_count() or 2) - 1)),
    )
    parser.add_argument("--skip-factor-scan", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "major_revision_v9",
    )
    args = parser.parse_args()
    if args.mc_runs < 30:
        raise ValueError("Major-revision Monte Carlo requires at least 30 test seeds")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_has_methods(payload: dict, required_methods: set[str]) -> bool:
    if "rows" in payload:
        return required_methods.issubset({row["method"] for row in payload["rows"]})
    if "summary" in payload:
        return required_methods.issubset(set(payload["summary"]))
    return False


def load_completed_stage(
    path: Path,
    label: str,
    *,
    required_methods: set[str] | None = None,
) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if required_methods is not None and not _stage_has_methods(payload, required_methods):
        if "rows" in payload:
            missing = sorted(required_methods - {row["method"] for row in payload["rows"]})
        else:
            missing = sorted(required_methods - set(payload.get("summary", {})))
        print(
            f"  Cached {label} is stale and will be recomputed: "
            f"missing methods {missing}"
        )
        return None
    print(f"  Reusing completed {label}: {path}")
    return payload


def main() -> None:
    args = parse_args()
    training_seeds = tuple(int(value) for value in args.training_seeds.split(","))
    if len(training_seeds) < 5 or len(set(training_seeds)) != len(training_seeds):
        raise ValueError("Provide at least five distinct training seeds")

    project_dir = Path(__file__).resolve().parent
    output_dir = (project_dir / args.output_dir).resolve()
    model_dir = output_dir / "models"
    figure_dir = project_dir / "fig"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    config = make_config(args.mode)

    ensemble_path = output_dir / "training_ensemble.json"
    complete_ensemble = None
    if args.resume_models and ensemble_path.exists():
        candidate = json.loads(ensemble_path.read_text(encoding="utf-8"))
        candidate_records = candidate.get("models", [])
        if (
            tuple(record.get("training_seed") for record in candidate_records)
            == training_seeds
            and all(Path(record["model_path"]).exists() for record in candidate_records)
            and all(
                sha256_file(Path(record["model_path"])) == record["sha256"]
                for record in candidate_records
            )
        ):
            complete_ensemble = candidate_records

    if complete_ensemble is not None:
        print("[1/7] Reusing the fixed dataset specification and split indices...")
        print("[2/7] Loading the total-acceleration training ensemble...")
        records = complete_ensemble
        models = [load_model(Path(record["model_path"])) for record in records]
    else:
        print("[1/7] Building the fixed training/validation dataset...")
        features, labels = build_dataset(config, show_progress=True)
    if complete_ensemble is None and args.resume_models:
        print("[2/7] Resuming the total-acceleration training ensemble...")
        models, records = train_model_ensemble(
            config,
            features,
            labels,
            training_seeds,
            model_dir,
            show_progress=True,
            resume_existing=True,
        )
    elif complete_ensemble is None:
        print("[2/7] Training five independent total-acceleration LSTMs...")
        models, records = train_model_ensemble(
            config,
            features,
            labels,
            training_seeds,
            model_dir,
            show_progress=True,
        )
    save_json(
        {
            "experiment_version": "major-revision-v9",
            "method_focus": "Adaptive GM-AEKF",
            "data_seed": config.seed,
            "fixed_split_across_training_seeds": True,
            "training_seeds": list(training_seeds),
            "models": records,
        },
        ensemble_path,
    )

    validation_indices = records[0]["history"]["validation_indices"]
    validation_sequences = build_validation_sequences(config, validation_indices)
    print("[3/7] Tuning the expanded GM grid on validation trajectories only...")
    gm_path = output_dir / "gm_validation_tuning.json"
    gm_tuning = (
        load_completed_stage(gm_path, "GM validation search")
        if args.resume_results
        else None
    )
    if gm_tuning is None:
        gm_tuning = tune_gm_hyperparameters(
            validation_sequences,
            config,
            show_progress=True,
            workers=args.workers,
        )
        save_json(gm_tuning, gm_path)
    else:
        config.gm_tau_s = float(gm_tuning["selected_tau_s"])
        config.gm_drive_psd = float(gm_tuning["selected_drive_psd_m2_s5"])
    grid_status = gm_tuning["grid_diagnostics"]["drive_grid_status"]
    if grid_status == "unresolved_boundary_optimum":
        raise RuntimeError(
            "GM drive-PSD optimum still lies on an unresolved grid boundary; "
            "expand the validation grid before testing"
        )
    print("[3b/7] Tuning Adaptive GM-AEKF online parameters on validation trajectories...")
    adaptive_gm_path = output_dir / "adaptive_gm_validation_tuning.json"
    adaptive_gm_tuning = (
        load_completed_stage(adaptive_gm_path, "Adaptive GM validation search")
        if args.resume_results
        else None
    )
    if adaptive_gm_tuning is None:
        adaptive_gm_tuning = tune_adaptive_gm_hyperparameters(
            validation_sequences,
            config,
            show_progress=True,
        )
        save_json(adaptive_gm_tuning, adaptive_gm_path)
    else:
        apply_adaptive_gm_parameters(
            config,
            adaptive_gm_tuning["selected_parameters"],
        )

    residual_ensemble_path = output_dir / "residual_training_ensemble.json"
    residual_records = None
    if args.resume_models and residual_ensemble_path.exists():
        candidate = json.loads(residual_ensemble_path.read_text(encoding="utf-8"))
        candidate_records = candidate.get("models", [])
        if (
            tuple(record.get("training_seed") for record in candidate_records)
            == training_seeds
            and all(
                record.get("model_kind") == config.gm_lstm_target_mode
                for record in candidate_records
            )
            and all(Path(record["model_path"]).exists() for record in candidate_records)
            and all(
                sha256_file(Path(record["model_path"])) == record["sha256"]
                for record in candidate_records
            )
        ):
            residual_records = candidate_records

    if residual_records is not None:
        print("[4/7] Loading the dedicated GM-residual training ensemble...")
        residual_models = [
            load_model(Path(record["model_path"])) for record in residual_records
        ]
    else:
        print("[4/7] Building GM-aware labels and training residual-injection LSTMs...")
        residual_features, residual_labels = build_gm_residual_dataset(
            config, show_progress=True
        )
        residual_models, residual_records = train_model_ensemble(
            config,
            residual_features,
            residual_labels,
            training_seeds,
            model_dir,
            show_progress=True,
            resume_existing=args.resume_models,
            model_kind=config.gm_lstm_target_mode,
        )
    calibration_sequences = validation_sequences[
        : min(config.residual_calibration_sequences, len(validation_sequences))
    ]
    for model, record in zip(residual_models, residual_records, strict=True):
        if "residual_injection_calibration" not in record["history"]:
            calibration = calibrate_gm_residual_injection(
                model, calibration_sequences, config
            )
            record["history"]["residual_injection_calibration"] = calibration
            model_path = Path(record["model_path"])
            save_model(model, model_path)
            record["sha256"] = sha256_file(model_path)
            model_path.with_suffix(".json").write_text(
                json.dumps(record, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
    save_json(
        {
            "experiment_version": "major-revision-v9",
            "method_focus": "Adaptive GM-AEKF",
            "target": (
                "a_true[k], converted online to a_true[k] - a_gm[k|k] "
                "for interval k -> k+1"
            ),
            "feature_source": "causal shadow GM-9D filter",
            "training_seeds": list(training_seeds),
            "models": residual_records,
        },
        residual_ensemble_path,
    )

    print("[5/7] Running crossed 5-training-seed x 30-test-seed ID experiment...")
    id_path = output_dir / "crossed_id.json"
    id_result = (
        load_completed_stage(
            id_path,
            "crossed ID experiment",
            required_methods=REQUIRED_CROSSED_METHODS,
        )
        if args.resume_results
        else None
    )
    if id_result is None:
        id_result = run_crossed_scenario_experiment(
            config,
            models,
            residual_models,
            training_seeds,
            scenario="nonlinear_structured",
            runs=args.mc_runs,
            seed_offset=2_000_000,
            include_ablation=True,
            workers=args.workers,
            show_progress=True,
        )
        save_json(id_result, id_path)
        save_crossed_rows_csv(id_result, output_dir / "crossed_id_rows.csv")

    if args.skip_factor_scan:
        factor_result = None
    else:
        print("[6/7] Running RMS-controlled single-factor boundary scans...")
        factor_path = output_dir / "single_factor_scan.json"
        factor_checkpoint_path = output_dir / "single_factor_scan.checkpoint.jsonl"
        if not args.resume_results:
            factor_checkpoint_path.unlink(missing_ok=True)
        factor_result = (
            load_completed_stage(
                factor_path,
                "single-factor boundary scan",
                required_methods=REQUIRED_FACTOR_METHODS,
            )
            if args.resume_results
            else None
        )
        if factor_result is None:
            factor_result = run_factor_scan(
                config,
                models,
                residual_models,
                training_seeds,
                runs=args.mc_runs,
                workers=args.workers,
                show_progress=True,
                checkpoint_path=(
                    factor_checkpoint_path if args.resume_results else None
                ),
            )
            save_json(factor_result, factor_path)
            save_factor_rows_csv(
                factor_result, output_dir / "single_factor_scan_rows.csv"
            )

    print("[7/7] Plotting and writing the reproducibility manifest...")
    plot_holm_ablation(id_result, figure_dir)
    plot_adaptive_gm_ablation(id_result, figure_dir)
    plot_consistency_coverage(id_result, figure_dir)
    plot_training_seed_summary(id_result, figure_dir)
    if factor_result is not None:
        plot_factor_boundaries(factor_result, figure_dir)
    source_files = [
        project_dir / "run_major_revision.py",
        project_dir / "audit_major_revision.py",
        project_dir / "generate_major_revision_report.py",
        project_dir / "requirements.txt",
        project_dir / "VERSION",
        *sorted((project_dir / "pcdh_lstm_aekf").glob("*.py")),
        *sorted((project_dir / "tests").glob("*.py")),
    ]
    save_json(
        _json_ready(
            {
                "experiment_version": "major-revision-v9",
                "one_command": (
                    "python run_major_revision.py --mode paper --mc-runs 30"
                ),
                "primary_method": "Adaptive GM-AEKF",
                "training_seeds": training_seeds,
                "test_seed_count": args.mc_runs,
                "process_noise_model": "continuous white acceleration sampled from Van Loan Qd",
                "acceleration_uncertainty": (
                    "validation residual second moment propagated as G Sigma_a G^T"
                ),
                "gm_tuning": gm_tuning,
                "adaptive_gm_tuning": adaptive_gm_tuning,
                "config": asdict(config),
                "models": records,
                "gm_residual_models": residual_records,
                "source_files_sha256": {
                    str(path.relative_to(project_dir)): sha256_file(path)
                    for path in source_files
                },
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "matplotlib": matplotlib.__version__,
                },
            }
        ),
        output_dir / "reproducibility_manifest.json",
    )
    print(f"Done: {output_dir}")


if __name__ == "__main__":
    main()
