import numpy as np


def train_model(
    model,
    features: np.ndarray,
    labels: np.ndarray,
    config,
    *,
    training_seed: int | None = None,
    show_progress: bool = False,
) -> dict[str, object]:
    """Train with a deterministic sequence-level train/validation split."""

    samples = features.shape[0]
    if samples < 2:
        raise ValueError("At least two sequences are required for train/validation splitting")

    split_rng = np.random.default_rng(config.seed)
    rng = np.random.default_rng(config.seed if training_seed is None else training_seed)
    split_order = split_rng.permutation(samples)
    validation_count = int(round(samples * config.validation_fraction))
    validation_count = min(max(validation_count, 1), samples - 1)
    validation_indices = np.sort(split_order[:validation_count])
    training_indices = np.sort(split_order[validation_count:])
    training_feature_values = features[training_indices].reshape(-1, features.shape[-1])
    model.feature_mean = np.mean(training_feature_values, axis=0)
    feature_scale = np.std(training_feature_values, axis=0)
    model.feature_scale = np.where(feature_scale > 1e-6, feature_scale, 1.0)

    training_loss: list[float] = []
    validation_loss: list[float] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_parameters = {name: value.copy() for name, value in model.parameters.items()}

    for epoch in range(config.epochs):
        window_steps = int(getattr(config, "training_window_steps", 0))
        windows_per_sequence = int(
            getattr(config, "training_windows_per_sequence", 1)
        )
        use_windows = 0 < window_steps < features.shape[1]
        if use_windows:
            payloads = [
                (int(index), int(rng.integers(0, features.shape[1] - window_steps + 1)))
                for index in training_indices
                for _ in range(max(windows_per_sequence, 1))
            ]
        else:
            payloads = [(int(index), 0) for index in training_indices]
        order = rng.permutation(len(payloads))
        epoch_loss = 0.0
        epoch_samples = 0
        for start in range(0, order.size, config.batch_size):
            batch_payloads = [
                payloads[int(index)]
                for index in order[start:start + config.batch_size]
            ]
            if use_windows:
                feature_batch = np.stack(
                    [
                        features[index, offset:offset + window_steps]
                        for index, offset in batch_payloads
                    ]
                )
                label_batch = np.stack(
                    [
                        labels[index, offset:offset + window_steps]
                        for index, offset in batch_payloads
                    ]
                )
            else:
                indices = np.asarray(
                    [payload[0] for payload in batch_payloads], dtype=int
                )
                feature_batch = features[indices]
                label_batch = labels[indices]
            loss = model.train_batch(
                feature_batch,
                label_batch,
                config.learning_rate,
                config.gradient_clip_norm,
            )
            epoch_loss += loss * len(batch_payloads)
            epoch_samples += len(batch_payloads)

        train_value = epoch_loss / epoch_samples
        validation_value = model.evaluate_loss(
            features[validation_indices], labels[validation_indices]
        )["total"]
        training_loss.append(float(train_value))
        validation_loss.append(float(validation_value))
        if validation_value < best_validation_loss:
            best_validation_loss = float(validation_value)
            best_epoch = epoch + 1
            best_parameters = {name: value.copy() for name, value in model.parameters.items()}
        if show_progress and (
            epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == config.epochs
        ):
            print(
                f"  Training epoch {epoch + 1}/{config.epochs}: "
                f"train={train_value:.6f}, validation={validation_value:.6f}, "
                f"best_epoch={best_epoch}"
            )
        patience = int(getattr(config, "early_stopping_patience", 0))
        if patience > 0 and epoch + 1 - best_epoch >= patience:
            if show_progress:
                print(
                    f"  Early stopping at epoch {epoch + 1}; "
                    f"best epoch was {best_epoch}"
                )
            break

    for name, value in best_parameters.items():
        model.parameters[name][...] = value

    validation_acceleration, _, _, _ = model.forward(features[validation_indices])
    target_validation_acceleration = labels[validation_indices, :, :3]
    if model.output_mode == "gm_residual":
        flat_prediction = validation_acceleration.reshape(-1, 3)
        flat_target = target_validation_acceleration.reshape(-1, 3)
        numerator = np.sum(flat_prediction * flat_target, axis=0)
        denominator = np.sum(np.square(flat_prediction), axis=0)
        gain = np.divide(numerator, np.maximum(denominator, 1e-30))
        model.acceleration_output_gain = np.clip(gain, 0.0, 1.0)
        validation_acceleration = (
            validation_acceleration * model.acceleration_output_gain
        )
        acceleration_residual_tensor = (
            target_validation_acceleration - validation_acceleration
        )
        zero_target = target_validation_acceleration
    elif model.output_mode == "gm_total_acceleration":
        gm_acceleration = features[validation_indices, :, -9:-6] * model.a_max
        candidate_residual = validation_acceleration - gm_acceleration
        target_residual = target_validation_acceleration - gm_acceleration
        flat_candidate = candidate_residual.reshape(-1, 3)
        flat_target = target_residual.reshape(-1, 3)
        numerator = np.sum(flat_candidate * flat_target, axis=0)
        denominator = np.sum(np.square(flat_candidate), axis=0)
        gain = np.divide(numerator, np.maximum(denominator, 1e-30))
        model.residual_injection_gain = np.clip(gain, 0.0, 1.0)
        acceleration_residual_tensor = (
            target_residual
            - candidate_residual * model.residual_injection_gain
        )
        zero_target = target_residual
    else:
        acceleration_residual_tensor = (
            target_validation_acceleration - validation_acceleration
        )
        zero_target = target_validation_acceleration
    acceleration_residual = acceleration_residual_tensor.reshape(-1, 3)
    zero_residual_mse = float(np.mean(np.square(zero_target)))
    learned_residual_mse = float(np.mean(np.square(acceleration_residual)))
    acceleration_error_covariance = (
        acceleration_residual.T @ acceleration_residual
        / max(acceleration_residual.shape[0], 1)
    )
    acceleration_error_covariance = 0.5 * (
        acceleration_error_covariance + acceleration_error_covariance.T
    )
    eigenvalues, eigenvectors = np.linalg.eigh(acceleration_error_covariance)
    eigenvalues = np.maximum(eigenvalues, config.acceleration_covariance_floor)
    acceleration_error_covariance = (
        eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    )
    model.acceleration_error_covariance = acceleration_error_covariance

    return {
        "training_loss": training_loss,
        "validation_loss": validation_loss,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "epochs_completed": len(training_loss),
        "training_indices": training_indices.tolist(),
        "validation_indices": validation_indices.tolist(),
        "training_sequences": int(training_indices.size),
        "validation_sequences": int(validation_indices.size),
        "training_seed": int(config.seed if training_seed is None else training_seed),
        "training_window_steps": int(getattr(config, "training_window_steps", 0)),
        "training_windows_per_sequence": int(
            getattr(config, "training_windows_per_sequence", 1)
        ),
        "split_seed": int(config.seed),
        "acceleration_error_covariance_m2_s4": acceleration_error_covariance.tolist(),
        "acceleration_error_covariance_source": (
            "validation residual second moment after restoring the best epoch"
        ),
        "acceleration_output_gain": model.acceleration_output_gain.tolist(),
        "residual_injection_gain": model.residual_injection_gain.tolist(),
        "validation_learned_to_zero_mse_ratio": (
            learned_residual_mse / max(zero_residual_mse, 1e-30)
        ),
    }
