import numpy as np
from scipy.stats import chi2


def compute_metrics(
    sequence: dict[str, np.ndarray],
    result: dict[str, np.ndarray | float],
    *,
    start_index: int = 0,
    end_index: int | None = None,
) -> dict:
    """Compute accuracy, consistency, whiteness and runtime metrics."""

    steps = int(sequence["truth"].shape[0])
    start_index = int(np.clip(start_index, 0, steps))
    end_index = steps if end_index is None else int(np.clip(end_index, 0, steps))
    if end_index <= start_index:
        raise ValueError("Metric interval must contain at least one sample")
    interval = slice(start_index, end_index)

    error = sequence["truth"][interval] - result["estimate"][interval]
    position_rmse = float(np.sqrt(np.mean(np.sum(np.square(error[:, :3]), axis=1))))
    velocity_rmse = float(np.sqrt(np.mean(np.sum(np.square(error[:, 3:]), axis=1))))
    state_rmse = np.sqrt(np.mean(np.square(error), axis=0))

    interval_nis = result["nis"][interval]
    interval_dof = result["dof"][interval]
    valid_nis = np.isfinite(interval_nis) & (interval_dof > 0)
    nis_values = interval_nis[valid_nis]
    nis_dof = interval_dof[valid_nis]
    nis_lower = chi2.ppf(0.025, nis_dof)
    nis_upper = chi2.ppf(0.975, nis_dof)
    nis_coverage = float(np.mean((nis_values >= nis_lower) & (nis_values <= nis_upper)))
    normalized_nis = float(np.sum(nis_values) / np.sum(nis_dof))

    interval_nees = result["nees"][interval]
    nees_values = interval_nees[np.isfinite(interval_nees)]
    nees_lower = chi2.ppf(0.025, 6)
    nees_upper = chi2.ppf(0.975, 6)
    nees_coverage = float(np.mean((nees_values >= nees_lower) & (nees_values <= nees_upper)))
    normalized_nees = float(np.mean(nees_values) / 6.0)

    lag_correlations = []
    normalized_innovation = result["features"][interval, :3]
    interval_mask = sequence["mask"][interval]
    for channel in range(3):
        x0 = normalized_innovation[:-1, channel]
        x1 = normalized_innovation[1:, channel]
        valid = interval_mask[:-1, channel] & interval_mask[1:, channel]
        if valid.sum() > 3 and np.std(x0[valid]) > 1e-12 and np.std(x1[valid]) > 1e-12:
            lag_correlations.append(abs(float(np.corrcoef(x0[valid], x1[valid])[0, 1])))

    return {
        "position_rss_rmse_m": position_rmse,
        "velocity_rss_rmse_mps": velocity_rmse,
        "state_rmse": state_rmse.tolist(),
        "normalized_nis": normalized_nis,
        "normalized_nees": normalized_nees,
        "nis_95_coverage": nis_coverage,
        "nees_95_coverage": nees_coverage,
        "mean_abs_lag1_innovation_corr": float(np.mean(lag_correlations)) if lag_correlations else None,
        "runtime_ms_per_step": float(result["runtime_ms_per_step"]),
        "finite": bool(np.isfinite(result["estimate"]).all()),
        "start_index": start_index,
        "end_index": end_index,
        "sample_count": end_index - start_index,
    }


def compute_segmented_metrics(
    sequence: dict[str, np.ndarray],
    result: dict[str, np.ndarray | float],
    config,
) -> dict[str, dict]:
    """Compute full, post-burn-in, and post-burn-in three-stage metrics."""

    steps = int(sequence["truth"].shape[0])
    burn_in_steps = min(
        int(round(config.burn_in_seconds / config.dt)),
        max(0, steps - 3),
    )
    remaining = steps - burn_in_steps
    first_end = burn_in_steps + remaining // 3
    second_end = burn_in_steps + 2 * remaining // 3
    intervals = {
        "full": (0, steps),
        "post_burn_in": (burn_in_steps, steps),
        "early": (burn_in_steps, max(first_end, burn_in_steps + 1)),
        "middle": (max(first_end, burn_in_steps + 1), max(second_end, first_end + 1)),
        "late": (max(second_end, first_end + 1), steps),
    }
    return {
        name: compute_metrics(
            sequence,
            result,
            start_index=start,
            end_index=end,
        )
        for name, (start, end) in intervals.items()
    }
