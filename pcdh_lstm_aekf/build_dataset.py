import numpy as np

from .cw_matrices import cw_matrices
from .run_fixed_ekf import run_fixed_ekf
from .run_gm_ekf import run_gm_9d_ekf
from .simulate_sequence import simulate_sequence


TRAINING_SCENARIOS = (
    "gaussian",
    "piecewise",
    "train",
    "dropout",
    "nonlinear",
    "nonlinear_shadow",
    "nonlinear_pulse",
    "nonlinear_state_drag",
    "nonlinear_structured",
    "nonlinear_structured",
    "nonlinear_structured",
    "nonlinear_structured",
    "nonlinear_structured",
    "nonlinear_structured",
    "nonlinear_structured",
    "nonlinear_structured_legacy",
)


def training_sequence_spec(config, index: int) -> tuple[int, str]:
    """Return the canonical seed and scenario for one complete training sequence."""

    return config.seed + 101 * index, TRAINING_SCENARIOS[index % len(TRAINING_SCENARIOS)]


def build_validation_sequences(config, validation_indices: list[int]) -> list[dict[str, object]]:
    """Regenerate the exact complete trajectories reserved for validation."""

    sequences = []
    for index in validation_indices:
        seed, scenario = training_sequence_spec(config, index)
        sequences.append(simulate_sequence(config, seed, scenario))
    return sequences


def build_dataset(config, *, show_progress: bool = False) -> tuple[np.ndarray, np.ndarray]:
    #生成因果特征序列，以及下一时刻双输出标签。
    """Generate causal feature sequences and next-step dual-head labels."""

    #存放特征和标签的列表
    feature_batches = []
    label_batches = []
    for index in range(config.train_sequences):
        #生成一条仿真轨迹，每条序列的环境不同，保证覆盖多种环境
        seed, scenario = training_sequence_spec(config, index)
        sequence = simulate_sequence(config, seed, scenario)
        #运行固定EKF，获取特征
        baseline = run_fixed_ekf(sequence, config)
        #生成下一时刻索引，但是不超过序列长度，保证标签数据的正确性
        next_index = np.minimum(np.arange(config.steps) + 1, config.steps - 1)
        #真实的过程噪声，使用真实的过程噪声和观测噪声作为标签数据，保证标签数据的正确性 
        log_q = np.log10(sequence["q_psd"] / config.q0_psd[None, :])
        #预测下一步的R
        log_r_next = np.log10(sequence["r_var"][next_index] / config.r0_var[None, :])
        #构造标签：未建模加速度（3维）、过程噪声变化（3维）和量测噪声（3维）
        labels = np.concatenate((sequence["acceleration"], log_q, log_r_next), axis=1)
        #保存数据
        feature_batches.append(baseline["features"])
        label_batches.append(labels)
        if show_progress and (
            index == 0 or (index + 1) % 40 == 0 or index + 1 == config.train_sequences
        ):
            print(f"  Dataset progress: {index + 1}/{config.train_sequences} sequences")
    #输出训练数据和标签数据，分别是特征和下一时刻的双输出标签。
    return np.stack(feature_batches), np.stack(label_batches)
    #如果训练序列：20条。每条：500步。
    #那么输出的训练数据和标签数据的形状分别是：(20, 500, 15)和(20, 500, 9)


def build_gm_residual_features(gm_result: dict[str, object], config) -> np.ndarray:
    """Build causal features for the dedicated post-GM residual learner.

    The first 15 channels are the GM filter's own innovation/covariance features.
    The remaining channels expose the estimated navigation state, acceleration
    state, its causal first difference, and its posterior uncertainty.  No truth
    quantity enters the feature vector.
    """

    base = np.asarray(gm_result["features"], dtype=float)
    acceleration = np.asarray(gm_result["acceleration_estimate"], dtype=float)
    acceleration_covariance = np.asarray(
        gm_result["acceleration_covariance"], dtype=float
    )
    acceleration_scale = np.maximum(np.asarray(config.a_max, dtype=float), 1e-12)
    normalized_acceleration = acceleration / acceleration_scale
    navigation_state = np.asarray(gm_result["estimate"], dtype=float)
    state_scale = np.concatenate(
        (
            np.maximum(np.abs(np.asarray(config.initial_state[:3])), 100.0),
            np.maximum(np.abs(np.asarray(config.initial_state[3:])), 0.05),
        )
    )
    normalized_navigation_state = navigation_state / state_scale
    continuous_matrix, _, _ = cw_matrices(config.mean_motion, config.dt)
    kinematic_residual = np.zeros_like(acceleration)
    if acceleration.shape[0] > 1:
        observed_acceleration = (
            navigation_state[1:, 3:] - navigation_state[:-1, 3:]
        ) / config.dt
        modeled_acceleration = (
            navigation_state[:-1] @ continuous_matrix[3:, :].T
        )
        kinematic_residual[1:] = (
            observed_acceleration - modeled_acceleration - acceleration[:-1]
        )
    normalized_kinematic_residual = np.clip(
        kinematic_residual / acceleration_scale, -6.0, 6.0
    )
    smoothed_kinematic_residual = np.zeros_like(normalized_kinematic_residual)
    kinematic_beta = 0.90
    for index in range(1, acceleration.shape[0]):
        smoothed_kinematic_residual[index] = (
            kinematic_beta * smoothed_kinematic_residual[index - 1]
            + (1.0 - kinematic_beta) * normalized_kinematic_residual[index]
        )
    acceleration_difference = np.zeros_like(acceleration)
    acceleration_difference[1:] = acceleration[1:] - acceleration[:-1]
    normalized_difference = acceleration_difference / acceleration_scale
    stationary_variance = max(
        float(config.gm_drive_psd * config.gm_tau_s / 2.0), 1e-18
    )
    normalized_uncertainty = np.log10(
        np.maximum(
            np.diagonal(acceleration_covariance, axis1=1, axis2=2)
            / stationary_variance,
            1e-12,
        )
    )
    if "time" in gm_result:
        time = np.asarray(gm_result["time"], dtype=float)
    else:
        time = np.arange(acceleration.shape[0], dtype=float) * config.dt
    duration = max(float(time[-1] - time[0]) if time.size > 1 else config.dt, config.dt)
    progress = ((time - float(time[0])) / duration).reshape(-1, 1)
    mean_motion = float(gm_result.get("mean_motion", config.mean_motion))
    orbital_phase = mean_motion * time
    short_phase = 2.0 * np.pi * time / max(0.075 * duration, config.dt)
    time_features = np.column_stack(
        (
            progress[:, 0],
            np.sin(orbital_phase),
            np.cos(orbital_phase),
            np.sin(2.0 * orbital_phase),
            np.sin(short_phase),
            np.cos(short_phase),
        )
    )
    return np.concatenate(
        (
            base,
            time_features,
            normalized_navigation_state,
            normalized_kinematic_residual,
            smoothed_kinematic_residual,
            normalized_acceleration,
            normalized_difference,
            normalized_uncertainty,
        ),
        axis=1,
    )


def build_gm_residual_dataset(
    config, *, show_progress: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Generate GM-aware features and acceleration-supervision labels.

    The default target is total unmodeled acceleration because it is more
    observable than the small post-GM remainder.  The filter converts it to a
    residual by subtracting the posterior GM acceleration state.  Direct
    residual supervision remains available as an ablation mode.
    """

    feature_batches = []
    label_batches = []
    for index in range(config.train_sequences):
        seed, scenario = training_sequence_spec(config, index)
        sequence = simulate_sequence(config, seed, scenario)
        gm_result = run_gm_9d_ekf(sequence, config, adapt_r=True)
        features = build_gm_residual_features(gm_result, config)
        if config.gm_lstm_target_mode == "gm_residual":
            acceleration_target = (
                np.asarray(sequence["acceleration"], dtype=float)
                - np.asarray(gm_result["acceleration_estimate"], dtype=float)
            )
        elif config.gm_lstm_target_mode == "gm_total_acceleration":
            acceleration_target = np.asarray(sequence["acceleration"], dtype=float)
        else:
            raise ValueError(
                f"Unknown gm_lstm_target_mode: {config.gm_lstm_target_mode}"
            )
        next_index = np.minimum(np.arange(config.steps) + 1, config.steps - 1)
        log_q = np.log10(sequence["q_psd"] / config.q0_psd[None, :])
        log_r_next = np.log10(
            sequence["r_var"][next_index] / config.r0_var[None, :]
        )
        labels = np.concatenate((acceleration_target, log_q, log_r_next), axis=1)
        feature_batches.append(features)
        label_batches.append(labels)
        if show_progress and (
            index == 0 or (index + 1) % 40 == 0 or index + 1 == config.train_sequences
        ):
            print(
                f"  GM residual dataset progress: {index + 1}/"
                f"{config.train_sequences} sequences"
            )
    return np.stack(feature_batches), np.stack(label_batches)
