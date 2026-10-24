import numpy as np


IN_DISTRIBUTION_FAMILY = "randomized_sine_single_step"
MULTI_HARMONIC_PULSE_FAMILY = "multi_harmonic_pulse"
SHADOW_ECLIPSE_SWITCH_FAMILY = "shadow_eclipse_switch"
STATE_DEPENDENT_DRAG_FAMILY = "state_dependent_drag_like"
STRUCTURED_NONSTATIONARY_FAMILY = "structured_shadow_pulse_drag"
LEARNABLE_RESIDUAL_STRESS_FAMILY = "learnable_residual_stress"
DISTURBANCE_RANDOMIZATION_VERSION = "trajectory-domain-randomization-v4"


def _random_signed_vector(
    rng: np.random.Generator,
    nominal_abs: np.ndarray,
    scale_range: tuple[float, float],
) -> np.ndarray:
    magnitudes = nominal_abs * rng.uniform(*scale_range, size=3)
    signs = rng.choice(np.array([-1.0, 1.0]), size=3)
    return magnitudes * signs


def _copy_profile_value(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, list):
        return [
            {
                key: _copy_profile_value(item_value)
                for key, item_value in item.items()
            }
            if isinstance(item, dict)
            else _copy_profile_value(item)
            for item in value
        ]
    return value


def _scale_profile_amplitude(profile: dict[str, object], level: float) -> None:
    for key in (
        "amplitudes_mps2",
        "step_biases_mps2",
        "event_amplitudes_mps2",
        "state_gate_amplitudes_mps2",
    ):
        if key in profile:
            profile[key] = np.asarray(profile[key]) * level
    for container_key in ("components", "base_components"):
        for component in profile.get(container_key, []):
            _scale_profile_amplitude(component, level)


def _scale_profile_frequency(profile: dict[str, object], level: float) -> None:
    for key in ("omega_start_radps", "omega_rate_radps2", "event_ring_omega_radps"):
        if key in profile:
            profile[key] = np.asarray(profile[key]) * level
    if "micro_pulse_period_s" in profile and level > 0.0:
        profile["micro_pulse_period_s"] = float(profile["micro_pulse_period_s"]) / level
    for container_key in ("components", "base_components"):
        for component in profile.get(container_key, []):
            _scale_profile_frequency(component, level)


def _set_profile_transition_duration(profile: dict[str, object], level: float) -> None:
    if "step_times_s" in profile:
        profile["step_transition_durations_s"] = np.full(
            np.asarray(profile["step_times_s"]).size,
            level,
            dtype=float,
        )
    if "event_rise_s" in profile:
        profile["event_rise_s"] = np.full(
            np.asarray(profile["event_rise_s"]).size,
            max(level, 0.05),
            dtype=float,
        )
    if "event_decay_s" in profile:
        profile["event_decay_s"] = np.full(
            np.asarray(profile["event_decay_s"]).size,
            max(2.0 * level, 0.5),
            dtype=float,
        )
    if "micro_pulse_width_s" in profile:
        profile["micro_pulse_width_s"] = max(level, 0.05)
    for container_key in ("components", "base_components"):
        for component in profile.get(container_key, []):
            _set_profile_transition_duration(component, level)


def sample_acceleration_profile(
    config,
    rng: np.random.Generator,
    duration_s: float,
    family: str,
    severity: float = 1.0,
) -> dict[str, object]:
    """Sample one reproducible trajectory-level unmodeled-acceleration profile."""

    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")

    amplitudes = severity * config.acceleration_nominal_amplitude * rng.uniform(
        *config.acceleration_amplitude_scale_range, size=3
    )
    phases = rng.uniform(0.0, 2.0 * np.pi, size=3)

    if family == IN_DISTRIBUTION_FAMILY:
        omega_start = config.acceleration_nominal_angular_frequency * rng.uniform(
            *config.acceleration_frequency_scale_range, size=3
        )
        omega_rate = np.zeros(3, dtype=float)
        step_times = np.array(
            [duration_s * rng.uniform(*config.acceleration_step_fraction_range)],
            dtype=float,
        )
        step_biases = severity * _random_signed_vector(
            rng,
            config.acceleration_nominal_step_abs,
            config.acceleration_step_scale_range,
        )[None, :]
    elif family == MULTI_HARMONIC_PULSE_FAMILY:
        omega_start = config.acceleration_nominal_angular_frequency * rng.uniform(
            0.8, 1.4, size=3
        )
        omega_rate = np.zeros(3, dtype=float)
        pulse_count = 3
        step_times = np.sort(rng.uniform(0.18, 0.82, size=pulse_count) * duration_s)
        step_biases = severity * np.stack(
            [
                _random_signed_vector(
                    rng,
                    1.6 * config.acceleration_nominal_step_abs,
                    (0.55, 1.25),
                )
                for _ in range(pulse_count)
            ],
            axis=0,
        )
        harmonic_multipliers = np.array([1.0, 2.7, 5.3], dtype=float)
        harmonic_weights = rng.uniform(0.18, 0.42, size=(2, 3))
    elif family == SHADOW_ECLIPSE_SWITCH_FAMILY:
        omega_start = config.acceleration_nominal_angular_frequency * rng.uniform(
            0.45, 0.9, size=3
        )
        omega_rate = np.zeros(3, dtype=float)
        step_times = np.sort(rng.uniform(0.25, 0.75, size=2) * duration_s)
        step_biases = severity * np.stack(
            [
                _random_signed_vector(
                    rng,
                    2.4 * config.acceleration_nominal_step_abs,
                    (0.65, 1.35),
                )
                for _ in range(2)
            ],
            axis=0,
        )
    elif family == STATE_DEPENDENT_DRAG_FAMILY:
        omega_start = config.acceleration_nominal_angular_frequency * rng.uniform(
            0.55, 1.1, size=3
        )
        omega_rate = np.zeros(3, dtype=float)
        step_times = np.array(
            [duration_s * rng.uniform(*config.acceleration_step_fraction_range)],
            dtype=float,
        )
        step_biases = severity * _random_signed_vector(
            rng,
            0.75 * config.acceleration_nominal_step_abs,
            config.acceleration_step_scale_range,
        )[None, :]
        drag_gain = severity * rng.uniform(1.2e-5, 3.0e-5)
        radial_gain = severity * rng.uniform(2.0e-7, 7.0e-7)
    elif family == STRUCTURED_NONSTATIONARY_FAMILY:
        component_severity = severity
        component_families = (
            SHADOW_ECLIPSE_SWITCH_FAMILY,
            MULTI_HARMONIC_PULSE_FAMILY,
            STATE_DEPENDENT_DRAG_FAMILY,
        )
        components = [
            sample_acceleration_profile(
                config,
                rng,
                duration_s,
                component_family,
                component_severity,
            )
            for component_family in component_families
        ]
        component_weights = np.array([0.65, 0.55, 0.85], dtype=float)
        omega_start = np.zeros(3, dtype=float)
        omega_rate = np.zeros(3, dtype=float)
        step_times = np.array([], dtype=float)
        step_biases = np.zeros((0, 3), dtype=float)
    elif family == LEARNABLE_RESIDUAL_STRESS_FAMILY:
        base_components = [
            sample_acceleration_profile(
                config,
                rng,
                duration_s,
                SHADOW_ECLIPSE_SWITCH_FAMILY,
                0.75 * severity,
            ),
            sample_acceleration_profile(
                config,
                rng,
                duration_s,
                STATE_DEPENDENT_DRAG_FAMILY,
                0.65 * severity,
            ),
        ]
        omega_start = config.acceleration_nominal_angular_frequency * rng.uniform(
            1.15, 1.85, size=3
        )
        omega_rate = rng.uniform(-0.45, 0.45, size=3) * omega_start / duration_s
        step_times = np.array([], dtype=float)
        step_biases = np.zeros((0, 3), dtype=float)
        event_fractions = np.array([0.18, 0.34, 0.52, 0.71, 0.86], dtype=float)
        event_times = duration_s * np.clip(
            event_fractions + rng.normal(0.0, 0.018, size=event_fractions.size),
            0.08,
            0.94,
        )
        event_axes = rng.normal(size=(event_times.size, 3))
        event_axes /= np.maximum(
            np.linalg.norm(event_axes, axis=1, keepdims=True), 1e-12
        )
        event_magnitudes = severity * rng.uniform(0.55, 1.10, size=event_times.size)
        event_amplitudes = (
            event_axes
            * event_magnitudes[:, None]
            * np.asarray(config.acceleration_nominal_amplitude)[None, :]
        )
        gate_vectors = rng.normal(size=(3, 6))
        gate_vectors /= np.maximum(
            np.linalg.norm(gate_vectors[:, :3], axis=1, keepdims=True), 1e-12
        )
        gate_thresholds = rng.uniform(-0.55, 0.55, size=3)
        gate_widths = rng.uniform(0.10, 0.22, size=3)
    else:
        raise ValueError(f"Unknown acceleration family: {family}")

    profile = {
        "family": family,
        "amplitudes_mps2": amplitudes,
        "phases_rad": phases,
        "omega_start_radps": omega_start,
        "omega_rate_radps2": omega_rate,
        "step_times_s": step_times,
        "step_biases_mps2": step_biases,
        "step_transition_durations_s": np.zeros(step_times.size, dtype=float),
    }
    if family == MULTI_HARMONIC_PULSE_FAMILY:
        profile["harmonic_multipliers"] = harmonic_multipliers
        profile["harmonic_weights"] = harmonic_weights
        profile["pulse_widths_s"] = rng.uniform(0.015, 0.045, size=step_times.size) * duration_s
    elif family == SHADOW_ECLIPSE_SWITCH_FAMILY:
        profile["step_transition_durations_s"] = rng.uniform(0.025, 0.075, size=2) * duration_s
        profile["shadow_period_s"] = duration_s * rng.uniform(0.32, 0.48)
        profile["shadow_duty_cycle"] = rng.uniform(0.34, 0.58)
        profile["shadow_transition_s"] = duration_s * rng.uniform(0.015, 0.035)
    elif family == STATE_DEPENDENT_DRAG_FAMILY:
        profile["drag_gain"] = float(drag_gain)
        profile["radial_gain"] = float(radial_gain)
    elif family == STRUCTURED_NONSTATIONARY_FAMILY:
        profile["components"] = components
        profile["component_weights"] = component_weights
        profile["description"] = (
            "Composite nonstationary disturbance: shadow/SRP switching, "
            "finite-duration pulse accelerations, and state-dependent drag-like "
            "terms. It is designed to violate a single fixed-tau Gauss-Markov "
            "acceleration assumption while remaining deterministic from seed."
        )
    elif family == LEARNABLE_RESIDUAL_STRESS_FAMILY:
        profile["base_components"] = base_components
        profile["base_component_weights"] = np.array([0.42, 0.55], dtype=float)
        profile["event_times_s"] = np.sort(event_times)
        profile["event_amplitudes_mps2"] = event_amplitudes
        profile["event_rise_s"] = rng.uniform(0.5, 1.8, size=event_times.size)
        profile["event_decay_s"] = rng.uniform(7.0, 22.0, size=event_times.size)
        profile["event_ring_omega_radps"] = rng.uniform(0.42, 0.95, size=event_times.size)
        profile["event_ring_phase_rad"] = rng.uniform(0.0, 2.0 * np.pi, size=event_times.size)
        profile["state_gate_vectors"] = gate_vectors
        profile["state_gate_thresholds"] = gate_thresholds
        profile["state_gate_widths"] = gate_widths
        profile["state_gate_amplitudes_mps2"] = (
            severity
            * rng.uniform(0.20, 0.48, size=(3, 3))
            * np.asarray(config.acceleration_nominal_amplitude)[None, :]
            * rng.choice(np.array([-1.0, 1.0]), size=(3, 3))
        )
        profile["micro_pulse_period_s"] = duration_s * rng.uniform(0.055, 0.085)
        profile["micro_pulse_width_s"] = rng.uniform(0.45, 1.10)
        profile["description"] = (
            "Distribution-internal learnable residual stress test: deterministic "
            "state gates, scheduled pulse-and-decay events, and chirped "
            "high-frequency components. These terms preserve causal "
            "reproducibility but violate the single fixed-tau GM acceleration "
            "model, leaving structured residuals for the LSTM."
        )
    return profile


def _smooth_box(time: float, center: float, width: float) -> float:
    if width <= 0.0:
        return float(time >= center)
    left = center - 0.5 * width
    right = center + 0.5 * width
    rise = np.clip((time - left) / max(width, 1e-12), 0.0, 1.0)
    fall = np.clip((right - time) / max(width, 1e-12), 0.0, 1.0)
    return float((rise**2 * (3.0 - 2.0 * rise)) * (fall**2 * (3.0 - 2.0 * fall)))


def unmodeled_acceleration(
    time: float,
    profile: dict[str, object],
    state: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate a sampled trajectory-level acceleration profile."""

    phase = (
        np.asarray(profile["phases_rad"])
        + np.asarray(profile["omega_start_radps"]) * time
        + 0.5 * np.asarray(profile["omega_rate_radps2"]) * time**2
    )
    acceleration = np.asarray(profile["amplitudes_mps2"]) * np.sin(phase)
    family = str(profile.get("family", ""))
    effective_family = (
        str(profile.get("base_family", family))
        if family.startswith("single_factor_")
        else family
    )
    if effective_family == STRUCTURED_NONSTATIONARY_FAMILY:
        acceleration = np.zeros(3, dtype=float)
        for component, weight in zip(
            profile["components"], profile["component_weights"], strict=True
        ):
            acceleration = acceleration + float(weight) * unmodeled_acceleration(
                time,
                component,
                state,
            )
        return acceleration
    if effective_family == LEARNABLE_RESIDUAL_STRESS_FAMILY:
        acceleration = np.zeros(3, dtype=float)
        for component, weight in zip(
            profile["base_components"], profile["base_component_weights"], strict=True
        ):
            acceleration = acceleration + float(weight) * unmodeled_acceleration(
                time,
                component,
                state,
            )

        phase = (
            np.asarray(profile["phases_rad"])
            + np.asarray(profile["omega_start_radps"]) * time
            + 0.5 * np.asarray(profile["omega_rate_radps2"]) * time**2
        )
        acceleration = acceleration + np.asarray(profile["amplitudes_mps2"]) * np.sin(phase)

        for event_time, event_amplitude, rise, decay, omega, phase0 in zip(
            np.asarray(profile["event_times_s"]),
            np.asarray(profile["event_amplitudes_mps2"]),
            np.asarray(profile["event_rise_s"]),
            np.asarray(profile["event_decay_s"]),
            np.asarray(profile["event_ring_omega_radps"]),
            np.asarray(profile["event_ring_phase_rad"]),
            strict=True,
        ):
            elapsed = time - float(event_time)
            if elapsed < 0.0:
                continue
            envelope = (1.0 - np.exp(-elapsed / max(float(rise), 1e-9))) * np.exp(
                -elapsed / max(float(decay), 1e-9)
            )
            acceleration = acceleration + event_amplitude * envelope * np.cos(
                float(omega) * elapsed + float(phase0)
            )

        if state is not None:
            state_vector = np.asarray(state, dtype=float)
            state_scale = np.array([800.0, 500.0, 250.0, 0.12, 0.08, 0.05])
            normalized_state = state_vector / state_scale
            gate_values = np.asarray(profile["state_gate_vectors"]) @ normalized_state
            gates = 1.0 / (
                1.0
                + np.exp(
                    -(
                        gate_values - np.asarray(profile["state_gate_thresholds"])
                    )
                    / np.asarray(profile["state_gate_widths"])
                )
            )
            acceleration = acceleration + gates @ np.asarray(
                profile["state_gate_amplitudes_mps2"]
            )

        period = max(float(profile["micro_pulse_period_s"]), 1e-9)
        local = time % period
        micro = _smooth_box(local, 0.5 * period, float(profile["micro_pulse_width_s"]))
        acceleration = acceleration + 0.22 * micro * np.asarray(profile["amplitudes_mps2"])
        return np.clip(acceleration, -1.35e-3, 1.35e-3)
    if effective_family == MULTI_HARMONIC_PULSE_FAMILY:
        harmonic_multipliers = np.asarray(profile["harmonic_multipliers"])
        harmonic_weights = np.asarray(profile["harmonic_weights"])
        for multiplier, weights in zip(harmonic_multipliers[1:], harmonic_weights, strict=True):
            acceleration = acceleration + (
                np.asarray(profile["amplitudes_mps2"])
                * weights
                * np.sin(
                    np.asarray(profile["phases_rad"]) * (1.0 + 0.13 * multiplier)
                    + np.asarray(profile["omega_start_radps"]) * multiplier * time
                )
            )
    elif effective_family == SHADOW_ECLIPSE_SWITCH_FAMILY:
        period = float(profile["shadow_period_s"])
        duty = float(profile["shadow_duty_cycle"])
        transition = float(profile["shadow_transition_s"])
        local = time % max(period, 1e-9)
        ingress = duty * period
        eclipse_weight = 1.0 - _smooth_box(local, 0.5 * ingress, ingress + transition)
        acceleration = acceleration * eclipse_weight
    elif effective_family == STATE_DEPENDENT_DRAG_FAMILY and state is not None:
        velocity = np.asarray(state, dtype=float)[3:6]
        position = np.asarray(state, dtype=float)[:3]
        speed = float(np.linalg.norm(velocity))
        radius = float(np.linalg.norm(position))
        drag = -float(profile["drag_gain"]) * speed * velocity
        radial = -float(profile["radial_gain"]) * position / max(radius, 1e-9)
        acceleration = acceleration + drag + radial
    step_times = np.asarray(profile["step_times_s"])
    step_biases = np.asarray(profile["step_biases_mps2"])
    if step_times.size:
        transition_durations = np.asarray(
            profile.get("step_transition_durations_s", np.zeros(step_times.size)),
            dtype=float,
        )
        for step_time, step_bias, transition_duration in zip(
            step_times, step_biases, transition_durations, strict=True
        ):
            if transition_duration <= 0.0:
                weight = float(time >= step_time)
            else:
                normalized = np.clip(
                    (time - (step_time - 0.5 * transition_duration))
                    / transition_duration,
                    0.0,
                    1.0,
                )
                weight = float(normalized**2 * (3.0 - 2.0 * normalized))
            acceleration = acceleration + weight * step_bias
    return acceleration


def build_single_factor_profile(
    base_profile: dict[str, object],
    time: np.ndarray,
    factor: str,
    level: float,
    *,
    rms_match: bool,
) -> dict[str, object]:
    """Modify one disturbance factor while retaining the seed-specific base profile."""

    if factor not in {"amplitude", "frequency", "transition_duration"}:
        raise ValueError(f"Unknown disturbance factor: {factor}")
    if level < 0.0:
        raise ValueError("Factor levels must be nonnegative")

    profile = {key: _copy_profile_value(value) for key, value in base_profile.items()}
    profile["base_family"] = str(base_profile["family"])
    profile["family"] = f"single_factor_{factor}"
    profile["factor"] = factor
    profile["factor_level"] = float(level)
    profile["rms_matched"] = bool(rms_match)

    if factor == "amplitude":
        _scale_profile_amplitude(profile, level)
    elif factor == "frequency":
        _scale_profile_frequency(profile, level)
    else:
        _set_profile_transition_duration(profile, level)

    base_samples = np.stack(
        [unmodeled_acceleration(float(value), base_profile) for value in time]
    )
    modified_samples = np.stack([unmodeled_acceleration(float(value), profile) for value in time])
    base_rms = np.sqrt(np.mean(np.square(base_samples), axis=0))
    modified_rms = np.sqrt(np.mean(np.square(modified_samples), axis=0))
    if rms_match:
        scale = base_rms / np.maximum(modified_rms, 1e-15)
        profile["amplitudes_mps2"] *= scale
        profile["step_biases_mps2"] *= scale[None, :]
        if "event_amplitudes_mps2" in profile:
            profile["event_amplitudes_mps2"] *= scale[None, :]
        if "state_gate_amplitudes_mps2" in profile:
            profile["state_gate_amplitudes_mps2"] *= scale[None, :]
        modified_samples = np.stack(
            [unmodeled_acceleration(float(value), profile) for value in time]
        )
        modified_rms = np.sqrt(np.mean(np.square(modified_samples), axis=0))

    profile["base_rms_mps2"] = base_rms
    profile["realized_rms_mps2"] = modified_rms
    profile["realized_rms_ratio"] = modified_rms / np.maximum(base_rms, 1e-15)
    return profile
