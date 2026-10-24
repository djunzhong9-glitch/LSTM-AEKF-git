import numpy as np

from .constrained_q import assemble_q_from_basis, build_q_basis
from .cw_matrices import cw_matrices
from .measurement_function import measurement_function
from .rk4_step import rk4_step
#导入未建模加速度（太阳光压力、大气阻力、地球非球形引力、推力误差）函数
from .unmodeled_acceleration import (
    IN_DISTRIBUTION_FAMILY,
    LEARNABLE_RESIDUAL_STRESS_FAMILY,
    MULTI_HARMONIC_PULSE_FAMILY,
    SHADOW_ECLIPSE_SWITCH_FAMILY,
    STATE_DEPENDENT_DRAG_FAMILY,
    STRUCTURED_NONSTATIONARY_FAMILY,
    build_single_factor_profile,
    sample_acceleration_profile,
    unmodeled_acceleration,
)


def sample_state_process_increment(
    rng: np.random.Generator,
    covariance: np.ndarray,
) -> np.ndarray:
    """Sample one zero-mean Gaussian increment from a symmetric PSD covariance."""

    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    if float(eigenvalues.min()) < -1e-12:
        raise ValueError("Process covariance is not positive semidefinite")
    factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))[None, :]
    return factor @ rng.standard_normal(covariance.shape[0])


def simulate_sequence(
    config,
    seed: int,
    scenario: str,
    *,
    acceleration_variant: dict[str, object] | None = None,
) -> dict[str, object]:
    """Generate one causal relative-navigation trajectory and sensor sequence."""
    #rng(Random Number Generator) 是随机数生成器的意思
    #np.random生成一个正太分布随机数
    #default_rng(seed) 是 NumPy 中的一个函数，用于创建一个默认的随机数生成器实例，并使用指定的种子(seed)进行初始化。种子是一个整数值，用于确定随机数生成器的初始状态，从而确保在相同的种子下，每次运行代码时生成的随机数序列都是相同的。
    seed_streams = np.random.SeedSequence(seed).spawn(5)
    structure_rng = np.random.default_rng(seed_streams[0])
    acceleration_rng = np.random.default_rng(seed_streams[1])
    initial_rng = np.random.default_rng(seed_streams[2])
    process_rng = np.random.default_rng(seed_streams[3])
    measurement_rng = np.random.default_rng(seed_streams[4])
    steps = config.steps
    #生成一个0到steps-1的数组，再乘以config.dt，得到一个时间数组，表示每个时间步的时间点。
    time = np.arange(steps, dtype=float) * config.dt
    #半长轴
    semimajor_axis = config.semimajor_axis
    if scenario == "low_orbit":
        semimajor_axis = 6.8e6
    elif scenario in {"high_orbit", "hard"}:
        semimajor_axis = 8.2e6
    #轨道角速度
    mean_motion = float(np.sqrt(config.mu / semimajor_axis**3))
    a_matrix, _, _ = cw_matrices(mean_motion, config.dt)
    q_basis = build_q_basis(a_matrix, config.dt)

    phase_q = structure_rng.uniform(0.0, 2.0 * np.pi, size=3)
    phase_r = structure_rng.uniform(0.0, 2.0 * np.pi, size=3)
    progress = np.linspace(0.0, 1.0, steps)
    log_q = 0.35 * np.sin(2.0 * np.pi * progress[:, None] + phase_q)
    log_r = 0.30 * np.cos(2.6 * np.pi * progress[:, None] + phase_r)

    if scenario == "gaussian":
        log_q[:] = 0.0
        log_r[:] = 0.0
    elif scenario in {"piecewise", "train"}:
        log_q[progress > 0.35] += np.array([0.55, -0.20, 0.35])
        log_r[progress > 0.58] += np.array([0.55, 0.25, -0.30])
    elif scenario == "student_t":
        log_q += 0.25
        log_r += 0.40
    #传感器漂移
    elif scenario == "bias_drift":
        log_r += 0.25
    #后期模型误差变大。
    elif scenario == "dropout":
        log_q[progress > 0.45] += 0.35
    elif scenario in {
        "nonlinear",
        "nonlinear_pulse",
        "nonlinear_shadow",
        "nonlinear_state_drag",
        "nonlinear_structured",
        "nonlinear_structured_legacy",
        "low_orbit",
        "high_orbit",
    }:
        log_q += 0.20
    elif scenario == "hard":
        log_q += np.array([0.75, 0.55, 0.60])
        log_r += np.array([0.75, 0.50, 0.50])
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    #限制噪声范围
    log_q = np.clip(log_q, config.log_scale_min, config.log_scale_max)
    log_r = np.clip(log_r, config.log_scale_min, config.log_scale_max)
    #计算真实的过程噪声功率谱密度和量测噪声方差
    q_psd = config.q0_psd[None, :] * np.power(10.0, log_q)
    r_var = config.r0_var[None, :] * np.power(10.0, log_r)
    #保存真实状态(x,y,z,vx,vy,vz)
    truth = np.zeros((steps, 6), dtype=float)
    #量测(ρ,θ,ϕ)
    measurements = np.zeros((steps, 3), dtype=float)
    #记录有没有量测丢失，1完好，0丢失。
    masks = np.ones((steps, 3), dtype=bool)
    #加速度
    acceleration = np.zeros((steps, 3), dtype=float)
    process_noise_state = np.zeros((steps, 6), dtype=float)
    #真实初始状态=理论状态+随即扰动
    truth[0] = config.initial_state + initial_rng.normal(
        0.0, [20.0, 20.0, 10.0, 0.02, 0.02, 0.01]
    )
    nonlinear = scenario in {
        "nonlinear",
        "nonlinear_pulse",
        "nonlinear_shadow",
        "nonlinear_state_drag",
        "nonlinear_structured",
        "nonlinear_structured_legacy",
        "hard",
    }
    severity = 1.35 if scenario == "hard" else 1.0
    acceleration_family_by_scenario = {
        "nonlinear_pulse": MULTI_HARMONIC_PULSE_FAMILY,
        "nonlinear_shadow": SHADOW_ECLIPSE_SWITCH_FAMILY,
        "nonlinear_state_drag": STATE_DEPENDENT_DRAG_FAMILY,
        "nonlinear_structured": LEARNABLE_RESIDUAL_STRESS_FAMILY,
        "nonlinear_structured_legacy": STRUCTURED_NONSTATIONARY_FAMILY,
    }
    acceleration_family = acceleration_family_by_scenario.get(
        scenario, IN_DISTRIBUTION_FAMILY
    )
    acceleration_profile = sample_acceleration_profile(
        config,
        acceleration_rng,
        float(time[-1]) if steps > 1 else config.dt,
        acceleration_family,
        severity,
    )
    if acceleration_variant is not None:
        acceleration_profile = build_single_factor_profile(
            acceleration_profile,
            time,
            str(acceleration_variant["factor"]),
            float(acceleration_variant["level"]),
            rms_match=bool(acceleration_variant.get("rms_match", False)),
        )
    bias = np.zeros(3, dtype=float)
    for k in range(steps - 1):
        #加入真实世界扰动
        acceleration[k] = unmodeled_acceleration(
            time[k],
            acceleration_profile,
            truth[k],
        )
        deterministic_next = rk4_step(
            truth[k], config.dt, mean_motion, semimajor_axis, config.mu,
            acceleration[k], nonlinear,
        )
        q_discrete = assemble_q_from_basis(q_psd[k], q_basis)
        process_noise_state[k + 1] = sample_state_process_increment(
            process_rng, q_discrete
        )
        truth[k + 1] = deterministic_next + process_noise_state[k + 1]
    acceleration[-1] = acceleration[-2]
    acceleration_profile["trajectory_acceleration_rms_mps2"] = np.sqrt(
        np.mean(np.square(acceleration), axis=0)
    )

    dropout_rate = config.dropout_probability if scenario in {"dropout", "hard"} else 0.0
    for k in range(steps):
        #偏置漂移，零点慢慢移动
        if scenario in {"bias_drift", "hard"}:
            bias += measurement_rng.normal(0.0, np.sqrt(r_var[k]) * 0.025)
        #t分布异常噪声，偶发大误差
        if scenario in {"student_t", "hard"}:
            noise = (
                measurement_rng.standard_t(4.0, size=3)
                / np.sqrt(2.0)
                * np.sqrt(r_var[k])
            )
        else:
            noise = measurement_rng.normal(0.0, np.sqrt(r_var[k]))
        #正常高斯噪声，普通传感器误差。
        if scenario == "train" and measurement_rng.random() < 0.015:
            noise *= 6.0
        measurements[k] = measurement_function(truth[k]) + noise + bias
        if dropout_rate > 0.0:
            masks[k] = measurement_rng.random(3) >= dropout_rate
            if not masks[k].any():
                masks[k, measurement_rng.integers(0, 3)] = True
        measurements[k, ~masks[k]] = np.nan

    initial_estimate = truth[0] + initial_rng.normal(0.0, np.sqrt(config.p0_diag))
    return {
        "seed": int(seed),
        "scenario": scenario,
        "rng_stream_policy": (
            "SeedSequence(seed).spawn(5): structure, acceleration, initial, process, measurement"
        ),
        "time": time,
        "truth": truth,
        "measurements": measurements,
        "mask": masks,
        "acceleration": acceleration,
        "process_noise_state": process_noise_state,
        "process_noise_model": "continuous_white_acceleration_van_loan",
        "q_psd": q_psd,
        "r_var": r_var,
        "q0_psd": config.q0_psd.copy(),
        "r0_var": config.r0_var.copy(),
        "initial_estimate": initial_estimate,
        "mean_motion": np.array(mean_motion),
        "semimajor_axis": np.array(semimajor_axis),
        "acceleration_profile": acceleration_profile,
    }
