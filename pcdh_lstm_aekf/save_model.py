from pathlib import Path

import numpy as np

from .dual_head_lstm import DualHeadLSTM


def save_model(model, output_path: Path) -> None:
    """Save NumPy LSTM weights and output scaling metadata."""

    np.savez_compressed(
        output_path,
        **model.parameters,
        a_max=model.a_max,
        max_log_scale=np.array(model.max_log_scale),
        hidden_size=np.array(model.hidden_size),
        input_size=np.array(model.input_size),
        output_mode=np.array(model.output_mode),
        feature_mean=model.feature_mean,
        feature_scale=model.feature_scale,
        loss_weight_acceleration=np.array(model.loss_weights["acceleration"]),
        loss_weight_covariance=np.array(model.loss_weights["covariance"]),
        loss_weight_smoothness=np.array(model.loss_weights["smoothness"]),
        loss_weight_acceleration_smoothness=np.array(
            model.loss_weights["acceleration_smoothness"]
        ),
        acceleration_error_covariance=model.acceleration_error_covariance,
        acceleration_output_gain=model.acceleration_output_gain,
        residual_injection_gain=model.residual_injection_gain,
    )


def load_model(input_path: Path) -> DualHeadLSTM:
    """Load an inference-ready model from a saved NumPy weight archive."""

    with np.load(input_path) as archive:
        output_mode = (
            str(archive["output_mode"].item())
            if "output_mode" in archive
            else "total_acceleration"
        )
        loss_weights = {
            "acceleration": float(archive["loss_weight_acceleration"])
            if "loss_weight_acceleration" in archive
            else 1.0,
            "covariance": float(archive["loss_weight_covariance"])
            if "loss_weight_covariance" in archive
            else 0.7,
            "smoothness": float(archive["loss_weight_smoothness"])
            if "loss_weight_smoothness" in archive
            else 0.02,
            "acceleration_smoothness": float(
                archive["loss_weight_acceleration_smoothness"]
            )
            if "loss_weight_acceleration_smoothness" in archive
            else 0.0,
        }
        model = DualHeadLSTM(
            hidden_size=int(archive["hidden_size"]),
            a_max=archive["a_max"],
            max_log_scale=float(archive["max_log_scale"]),
            seed=0,
            input_size=int(archive["input_size"]) if "input_size" in archive else 15,
            output_mode=output_mode,
            loss_weights=loss_weights,
        )
        for name in model.parameters:
            model.parameters[name][...] = archive[name]
        if "feature_mean" in archive:
            model.feature_mean = archive["feature_mean"].copy()
        if "feature_scale" in archive:
            model.feature_scale = archive["feature_scale"].copy()
        if "acceleration_error_covariance" in archive:
            model.acceleration_error_covariance = archive[
                "acceleration_error_covariance"
            ].copy()
        if "acceleration_output_gain" in archive:
            model.acceleration_output_gain = archive[
                "acceleration_output_gain"
            ].copy()
        if "residual_injection_gain" in archive:
            model.residual_injection_gain = archive[
                "residual_injection_gain"
            ].copy()
    return model
