from .config import ExperimentConfig


def make_config(mode: str) -> ExperimentConfig:
    """Build either a runnable quick configuration or the paper-scale setup."""

    config = ExperimentConfig()
    if mode == "paper":
        config.steps = 2500
        config.train_sequences = 320
        config.hidden_size = 128
        config.residual_hidden_size = 128
        config.batch_size = 16
        config.epochs = 60
        config.residual_epochs = 120
        config.residual_training_window_steps = 420
        config.residual_training_windows_per_sequence = 4
        config.residual_calibration_sequences = 48
        config.residual_early_stopping_patience = 20
        config.adaptive_gm_window = 34
        config.adaptive_gm_alpha_ema = 0.92
        config.adaptive_gm_tau_max_log_step = 0.014
        config.adaptive_gm_drive_max_log_step = 0.028
    return config
