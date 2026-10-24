from dataclasses import dataclass, field

import numpy as np


@dataclass
class ExperimentConfig:
    """All numerical and training settings used by the reproduction."""

    mu: float = 3.986004418e14
    semimajor_axis: float = 7.0873e6
    dt: float = 0.1
    steps: int = 500
    train_sequences: int = 20
    epochs: int = 12
    hidden_size: int = 48
    residual_hidden_size: int = 64
    #batch_size 是每次训练迭代中使用的样本数量，影响模型的训练速度和稳定性。
    batch_size: int = 5
    learning_rate: float = 2.0e-3
    residual_learning_rate: float = 1.2e-3
    residual_epochs: int = 20
    residual_training_window_steps: int = 200
    residual_training_windows_per_sequence: int = 2
    gm_lstm_target_mode: str = "gm_total_acceleration"
    residual_calibration_sequences: int = 12
    residual_early_stopping_patience: int = 12
    seed: int = 20260719
    validation_fraction: float = 0.20
    gradient_clip_norm: float = 1.0
    acceleration_covariance_floor: float = 1.0e-12
    burn_in_seconds: float = 25.0
    #ema_beta 是指数移动平均(Exponential Moving Average)的衰减因子，用于平滑训练过程中的指标变化。
    ema_beta: float = 0.92
    #nis_feedback_gain 是用于归一化创新平方(Normalized Innovation Squared, NIS)反馈的增益参数，控制NIS反馈在训练过程中的影响程度。
    nis_feedback_gain: float = 0.08
    #对数尺度上限和下限
    log_scale_min: float = -2.0
    log_scale_max: float = 2.0
    # R 使用较快的 NIS 反馈；Q 不再直接跟随 NIS，避免把量测噪声突变误判为过程噪声。
    r_ema_beta: float = 0.86
    r_max_log_step: float = 0.12
    r_cm_window: int = 20
    r_update_alpha: float = 0.30
    r_cm_weight: float = 0.70
    # Q 使用动态约束协方差匹配慢更新。窗口越长越平滑，alpha 越小越不容易被瞬时异常带偏。
    q_cm_window: int = 35
    q_max_log_step: float = 0.015
    q_log_scale_min: float = -1.2
    q_log_scale_max: float = 0.8
    # Q 的主估计来自 LSTM 监督头；协方差匹配只在其附近小幅校正，避免单轨迹 CM 把 Q 推到上界。
    q_lstm_ema_beta: float = 0.94
    q_lstm_update_alpha: float = 0.10
    q_cm_correction_weight: float = 0.12
    q_cm_log_correction_limit: float = 0.18
    # 创新一阶相关用于判断是否存在持续模型误差；白噪声式创新优先交给 R 处理。
    q_innovation_corr_threshold: float = 0.18
    # Traditional adaptive-filter baselines use the same physical Q/R bounds.
    baseline_cm_window: int = 20
    baseline_cm_alpha: float = 0.20
    baseline_sage_husa_forgetting: float = 0.98
    baseline_fading_min: float = 1.0
    baseline_fading_max: float = 3.0
    baseline_coupled_ema_beta: float = 0.92
    baseline_coupled_gain: float = 0.12
    baseline_coupled_max_log_step: float = 0.05
    # Measurement-channel dropout probability used by dropout/hard simulation scenarios.
    dropout_probability: float = 0.08
    # Per-trajectory domain randomization for the modeled-but-unobserved acceleration.
    acceleration_nominal_amplitude: np.ndarray = field(
        default_factory=lambda: np.array([7.0e-4, 5.5e-4, 3.5e-4], dtype=float)
    )
    acceleration_nominal_angular_frequency: np.ndarray = field(
        default_factory=lambda: np.array([0.045, 0.032, 0.057], dtype=float)
    )
    acceleration_nominal_step_abs: np.ndarray = field(
        default_factory=lambda: np.array([2.5e-4, 1.8e-4, 1.2e-4], dtype=float)
    )
    acceleration_amplitude_scale_range: tuple[float, float] = (0.65, 1.35)
    acceleration_frequency_scale_range: tuple[float, float] = (0.65, 1.35)
    acceleration_step_fraction_range: tuple[float, float] = (0.25, 0.75)
    acceleration_step_scale_range: tuple[float, float] = (0.50, 1.50)
    # Singer/Gauss-Markov 9D baseline.  The selected values are overwritten only
    # by validation-set tuning and then frozen for every test trajectory.
    gm_tau_s: float = 45.0
    gm_drive_psd: float = 8.0e-9
    gm_tau_candidates_s: tuple[float, ...] = (2.0, 5.0, 15.0, 45.0, 120.0, 360.0)
    gm_drive_psd_candidates: tuple[float, ...] = (
        5.0e-10,
        2.0e-9,
        8.0e-9,
        3.2e-8,
        1.28e-7,
        5.12e-7,
        2.048e-6,
    )
    # Adaptive GM-AEKF keeps the 9D acceleration-state model but changes its
    # correlation time and driving PSD online when the innovation sequence and
    # acceleration state indicate nonstationary dynamics.
    adaptive_gm_tau_min_s: float = 10.0
    adaptive_gm_tau_max_s: float = 120.0
    adaptive_gm_drive_min_factor: float = 0.40
    adaptive_gm_drive_max_factor: float = 4.0
    adaptive_gm_window: int = 28
    adaptive_gm_alpha_ema: float = 0.90
    adaptive_gm_activity_bias: float = -1.80
    adaptive_gm_activity_slope: float = 2.8
    adaptive_gm_corr_weight: float = 2.4
    adaptive_gm_nis_weight: float = 0.9
    adaptive_gm_accel_diff_weight: float = 0.75
    adaptive_gm_tau_max_log_step: float = 0.018
    adaptive_gm_drive_max_log_step: float = 0.035
    adaptive_gm_process_covariance_floor: float = 1.0e-16
    adaptive_gm_stationary_reset_weight: float = 0.03
    adaptive_gm_use_axiswise_drive: bool = False
    adaptive_gm_axis_drive_floor: float = 0.35
    adaptive_gm_axis_drive_ceiling: float = 2.8
    adaptive_gm_use_likelihood_bank: bool = True
    adaptive_gm_bank_models: tuple[tuple[float, float], ...] = (
        (120.0, 0.50),
        (80.0, 0.50),
        (45.0, 1.00),
        (25.0, 2.00),
        (15.0, 4.00),
    )
    adaptive_gm_bank_covariance_penalty: float = 0.015
    # The dedicated GM-residual network spends most of its capacity on the
    # deployable residual-acceleration target; R remains a secondary head.
    residual_loss_acceleration_weight: float = 3.0
    residual_loss_covariance_weight: float = 0.0
    residual_loss_smoothness_weight: float = 0.0
    residual_loss_acceleration_smoothness_weight: float = 0.04
    #a_max 是加速度的最大值，用于限制模型的输出。
    a_max: np.ndarray = field(
        default_factory=lambda: np.array([1.5e-3, 1.5e-3, 1.0e-3], dtype=float)
    )
    #基准过程噪声功率谱密度 三个元素代表 x方向过程噪声强度 y方向过程噪声强度 z方向过程噪声强度
    q0_psd: np.ndarray = field(
        default_factory=lambda: np.array([2.0e-7, 2.0e-7, 1.2e-7], dtype=float)
    )
    #量测噪声参数，np.deg2rad 将角度转换为弧度
    r0_var: np.ndarray = field(
        default_factory=lambda: np.square(
            np.array([0.2, np.deg2rad(0.024), np.deg2rad(0.024)], dtype=float)
        )
    )
    #初始协方差矩阵和参考协方差矩阵
    p0_diag: np.ndarray = field(
        default_factory=lambda: np.square(
            np.array([5.0, 5.0, 5.0, 0.05, 0.05, 0.05], dtype=float)
        )
    )
    #参考协方差尺度，用于归一化等，与p0_diag（初始化）不同
    p_ref: np.ndarray = field(
        default_factory=lambda: np.square(
            np.array([5.0, 5.0, 5.0, 0.05, 0.05, 0.05], dtype=float)
        )
    )
    #初始状态向量，包含位置和速度信息
    initial_state: np.ndarray = field(
        default_factory=lambda: np.array([600.0, -250.0, 120.0, 0.08, -0.03, 0.01], dtype=float)
    )

    @property
    #计算平均轨道角速度
    def mean_motion(self) -> float:
        return float(np.sqrt(self.mu / self.semimajor_axis**3))
