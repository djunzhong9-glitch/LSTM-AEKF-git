import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


CORE_METRICS = (
    "position_rss_rmse_m",
    "velocity_rss_rmse_mps",
    "normalized_nis",
    "normalized_nees",
    "nis_95_coverage",
    "nees_95_coverage",
)
EXPECTED_TRAINING_SEEDS = {20260719, 20260791, 20260863, 20260935, 20261007}
EXPECTED_ID_TEST_SEEDS = set(range(22260719, 22260749))
EXPECTED_FACTOR_TEST_SEEDS = set(range(23260719, 23260749))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_csv(path: Path, expected_rows: int, expected_test_seeds: set[int]) -> dict:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    nonfinite = []
    for row_index, row in enumerate(rows, start=2):
        for metric in CORE_METRICS:
            if not math.isfinite(float(row[metric])):
                nonfinite.append({"row": row_index, "metric": metric})
    training_seeds = {
        int(row["training_seed"]) for row in rows if row.get("training_seed")
    }
    test_seeds = {int(row["test_seed"]) for row in rows}
    return {
        "path": str(path),
        "row_count": len(rows),
        "expected_row_count": expected_rows,
        "row_count_ok": len(rows) == expected_rows,
        "nonfinite": nonfinite,
        "all_core_metrics_finite": not nonfinite,
        "test_seeds_ok": test_seeds == expected_test_seeds,
        "training_seeds": sorted(training_seeds),
        "training_seeds_ok": (
            not training_seeds or training_seeds == EXPECTED_TRAINING_SEEDS
        ),
        "methods": sorted({row["method"] for row in rows}),
        "phases": sorted({row["phase"] for row in rows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Major Revision raw artifacts")
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("artifacts/major_revision_v9")
    )
    args = parser.parse_args()
    root = args.artifact_dir.resolve()

    training = json.loads((root / "training_ensemble.json").read_text(encoding="utf-8"))
    model_checks = []
    for record in training["models"]:
        model_path = Path(record["model_path"])
        actual_hash = sha256_file(model_path)
        model_checks.append(
            {
                "training_seed": int(record["training_seed"]),
                "model_path": str(model_path),
                "expected_sha256": record["sha256"],
                "actual_sha256": actual_hash,
                "hash_ok": actual_hash == record["sha256"],
            }
        )

    gm = json.loads((root / "gm_validation_tuning.json").read_text(encoding="utf-8"))
    adaptive_gm = json.loads(
        (root / "adaptive_gm_validation_tuning.json").read_text(encoding="utf-8")
    )
    crossed_id = json.loads((root / "crossed_id.json").read_text(encoding="utf-8"))
    factors = json.loads((root / "single_factor_scan.json").read_text(encoding="utf-8"))
    holm = crossed_id["confirmatory_ablation_tests"]
    adaptive_holm = crossed_id["adaptive_gm_ablation_tests"]
    phase_count = 5
    expected_id_rows = (
        sum(len(rows) for rows in crossed_id["rows_by_method"].values()) * phase_count
    )
    expected_factor_rows = len(factors["rows"]) * phase_count
    csv_checks = {
        "id": audit_csv(
            root / "crossed_id_rows.csv", expected_id_rows, EXPECTED_ID_TEST_SEEDS
        ),
        "single_factor": audit_csv(
            root / "single_factor_scan_rows.csv", expected_factor_rows, EXPECTED_FACTOR_TEST_SEEDS
        ),
    }
    checks = {
        "training_seed_set_ok": {
            int(record["training_seed"]) for record in training["models"]
        }
        == EXPECTED_TRAINING_SEEDS,
        "all_model_hashes_ok": all(item["hash_ok"] for item in model_checks),
        "gm_test_set_unused": gm["test_set_used_for_selection"] is False,
        "adaptive_gm_test_set_unused": (
            adaptive_gm["test_set_used_for_selection"] is False
        ),
        "gm_internal_plateau_resolved": (
            gm["grid_diagnostics"]["drive_grid_status"]
            == "interior_near_optimal_plateau"
        ),
        "holm_family_size_matches_tests": holm["family_size"] == len(holm["tests"]),
        "adaptive_holm_family_size_matches_tests": (
            adaptive_holm["family_size"] == len(adaptive_holm["tests"])
        ),
        "holm_inference_unit_ok": holm["inference_unit"]
        == "training-seed mean paired difference over common test seeds",
        "adaptive_holm_inference_unit_ok": adaptive_holm["inference_unit"]
        == "test-seed paired difference over common trajectories",
        "adaptive_gm_in_crossed_summary": "Adaptive GM-AEKF"
        in crossed_id["summary"],
        "adaptive_gm_in_factor_summary": all(
            "Adaptive GM-AEKF" in level_summary
            for factor_summary in factors["summary"].values()
            for level_summary in factor_summary.values()
        ),
        "all_csv_checks_ok": all(
            item["row_count_ok"]
            and item["all_core_metrics_finite"]
            and item["test_seeds_ok"]
            and item["training_seeds_ok"]
            for item in csv_checks.values()
        ),
    }
    report = {
        "artifact_dir": str(root),
        "checks": checks,
        "passed": all(checks.values()),
        "models": model_checks,
        "gm": {
            "selected_tau_s": gm["selected_tau_s"],
            "selected_drive_psd_m2_s5": gm["selected_drive_psd_m2_s5"],
            "grid_status": gm["grid_diagnostics"]["drive_grid_status"],
        },
        "adaptive_gm": {
            "selected_name": adaptive_gm["selected_name"],
            "selected_score": adaptive_gm["selected_score"],
            "selected_parameters": adaptive_gm["selected_parameters"],
        },
        "holm": {
            "family_size": holm["family_size"],
            "rejected_at_0_05": sum(
                bool(test["reject_at_0_05"]) for test in holm["tests"]
            ),
        },
        "adaptive_holm": {
            "family_size": adaptive_holm["family_size"],
            "rejected_at_0_05": sum(
                bool(test["reject_at_0_05"]) for test in adaptive_holm["tests"]
            ),
        },
        "csv": csv_checks,
    }
    (root / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["checks"], ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
