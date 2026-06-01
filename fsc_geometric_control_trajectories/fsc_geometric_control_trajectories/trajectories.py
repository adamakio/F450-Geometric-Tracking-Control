"""Trajectory primitives for geometric control references."""

from dataclasses import dataclass
import math
from typing import Tuple


Vector3 = Tuple[float, float, float]
NUM_FINITE_CYCLES = 3


@dataclass(frozen=True)
class DesiredTrajectoryState:
    """Container for the desired flat outputs and their derivatives."""

    position: Vector3
    velocity: Vector3
    acceleration: Vector3
    jerk: Vector3
    snap: Vector3
    yaw: float
    yaw_rate: float
    yaw_acceleration: float


def hover(position: Vector3, yaw: float) -> DesiredTrajectoryState:
    """Return a stationary hover trajectory at a fixed pose."""
    zero = (0.0, 0.0, 0.0)
    return DesiredTrajectoryState(
        position=position,
        velocity=zero,
        acceleration=zero,
        jerk=zero,
        snap=zero,
        yaw=yaw,
        yaw_rate=0.0,
        yaw_acceleration=0.0,
    )


def yaw_spin(
    t: float,
    position: Vector3,
    initial_yaw: float,
    yaw_rate: float,
) -> DesiredTrajectoryState:
    """Return a stationary hover trajectory with constant yaw spin."""
    zero = (0.0, 0.0, 0.0)
    return DesiredTrajectoryState(
        position=position,
        velocity=zero,
        acceleration=zero,
        jerk=zero,
        snap=zero,
        yaw=initial_yaw + yaw_rate * t,
        yaw_rate=yaw_rate,
        yaw_acceleration=0.0,
    )


def elliptic_helix(t: float) -> DesiredTrajectoryState:
    """Return an elliptic helix ascending along z from [0, 0, 0.165]."""
    angular_rate = 0.5
    ellipse_a = 1.0
    ellipse_b = 2.0
    climb_rate = 0.1
    if abs(angular_rate) <= 1e-9:
        return hover((0.0, 0.0, 0.165), 0.0)

    duration = NUM_FINITE_CYCLES * (2.0 * math.pi / abs(angular_rate))
    if t >= duration:
        final_yaw = angular_rate * duration
        final_position = (
            0.0,
            0.0,
            0.165 + climb_rate * duration,
        )
        return hover(final_position, final_yaw)

    phase = angular_rate * t
    sin_phase = math.sin(phase)
    cos_phase = math.cos(phase)
    omega_2 = angular_rate ** 2
    omega_3 = angular_rate ** 3
    omega_4 = angular_rate ** 4

    return DesiredTrajectoryState(
        position=(
            ellipse_a * sin_phase,
            ellipse_b * (1.0 - cos_phase),
            0.165 + climb_rate * t,
        ),
        velocity=(
            ellipse_a * angular_rate * cos_phase,
            ellipse_b * angular_rate * sin_phase,
            climb_rate,
        ),
        acceleration=(
            -ellipse_a * omega_2 * sin_phase,
            ellipse_b * omega_2 * cos_phase,
            0.0,
        ),
        jerk=(
            -ellipse_a * omega_3 * cos_phase,
            -ellipse_b * omega_3 * sin_phase,
            0.0,
        ),
        snap=(
            ellipse_a * omega_4 * sin_phase,
            -ellipse_b * omega_4 * cos_phase,
            0.0,
        ),
        yaw=phase,
        yaw_rate=angular_rate,
        yaw_acceleration=0.0,
    )


def circle(
    t: float,
    radius: float,
    angular_velocity: float,
    level_height: float,
) -> DesiredTrajectoryState:
    """Return a planar circle that starts at [0, 0, level_height]."""
    start_position = (0.0, 0.0, level_height)
    if abs(angular_velocity) <= 1e-9:
        return hover(start_position, 0.0)

    duration = NUM_FINITE_CYCLES * (2.0 * math.pi / abs(angular_velocity))
    if t >= duration:
        return hover(start_position, 0.0)

    phase = angular_velocity * t
    sin_phase = math.sin(phase)
    cos_phase = math.cos(phase)
    omega_2 = angular_velocity ** 2
    omega_3 = angular_velocity ** 3
    omega_4 = angular_velocity ** 4

    return DesiredTrajectoryState(
        position=(
            radius * (1.0 - cos_phase),
            radius * sin_phase,
            level_height,
        ),
        velocity=(
            radius * angular_velocity * sin_phase,
            radius * angular_velocity * cos_phase,
            0.0,
        ),
        acceleration=(
            radius * omega_2 * cos_phase,
            -radius * omega_2 * sin_phase,
            0.0,
        ),
        jerk=(
            -radius * omega_3 * sin_phase,
            -radius * omega_3 * cos_phase,
            0.0,
        ),
        snap=(
            -radius * omega_4 * cos_phase,
            radius * omega_4 * sin_phase,
            0.0,
        ),
        yaw=phase,
        yaw_rate=angular_velocity,
        yaw_acceleration=0.0,
    )


def figure_eight(
    t: float,
    radius: float,
    angular_velocity: float,
    level_height: float,
) -> DesiredTrajectoryState:
    """Return a planar figure-eight that starts at [0, 0, level_height]."""
    start_position = (0.0, 0.0, level_height)
    if abs(angular_velocity) <= 1e-9:
        return hover(start_position, 0.0)

    duration = NUM_FINITE_CYCLES * (2.0 * math.pi / abs(angular_velocity))
    if t >= duration:
        return hover(start_position, 0.0)

    phase = angular_velocity * t
    sin_phase = math.sin(phase)
    cos_phase = math.cos(phase)
    sin_double_phase = math.sin(2.0 * phase)
    cos_double_phase = math.cos(2.0 * phase)
    omega_2 = angular_velocity ** 2
    omega_3 = angular_velocity ** 3
    omega_4 = angular_velocity ** 4

    return DesiredTrajectoryState(
        position=(
            radius * sin_phase,
            0.5 * radius * sin_double_phase,
            level_height,
        ),
        velocity=(
            radius * angular_velocity * cos_phase,
            radius * angular_velocity * cos_double_phase,
            0.0,
        ),
        acceleration=(
            -radius * omega_2 * sin_phase,
            -2.0 * radius * omega_2 * sin_double_phase,
            0.0,
        ),
        jerk=(
            -radius * omega_3 * cos_phase,
            -4.0 * radius * omega_3 * cos_double_phase,
            0.0,
        ),
        snap=(
            radius * omega_4 * sin_phase,
            8.0 * radius * omega_4 * sin_double_phase,
            0.0,
        ),
        yaw=0.0,
        yaw_rate=0.0,
        yaw_acceleration=0.0,
    )
