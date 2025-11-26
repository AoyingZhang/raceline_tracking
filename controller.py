from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack

DT = 0.1

LOOKAHEAD_MIN = 10.0
LOOKAHEAD_GAIN = 0.6
LOOKAHEAD_MAX = 80.0
HAIRPIN_CURVATURE = 0.025
HAIRPIN_LOOKAHEAD_SCALE = 0.6
LOOKAHEAD_FINISH_FLOOR = 5.0
LAT_a_LIMIT = 15.0
MIN_SPEED_TARGET = 6.0
SPEED_FILTER_ALPHA = 0.25
HEADING_PENALTY_GAIN = 1.5
FUTURE_CURVATURE_STRIDE = 3
FUTURE_CURVATURE_STEPS = 20
CURVATURE_WEIGHT_DECAY = 0.15
LAP_START_DISTANCE = 25.0
FINISH_APPROACH_DISTANCE = 300.0
FINISH_TARGET_RADIUS = 40.0
MIN_CRUISE_SPEED = 8.0
FINISH_MIN_SPEED = 3.0
LAT_KP = 5.0
LAT_KI = 1.2
LAT_KD = 0.2
LONG_KP = 1.8
LONG_KI = 0.4
LONG_KD = 0.05

STEER_INTEGRAL_LIMIT = 0.5
SPEED_INTEGRAL_LIMIT = 12.0

@dataclass
class LoopMemory:
    integral: float = 0.0
    previous_error: float = 0.0


_lateral_loop = LoopMemory()
_longitudinal_loop = LoopMemory()
_speed_reference: Optional[float] = None
hasStarted = False


def _ensure_track_cache(racetrack: RaceTrack) -> None:
    if hasattr(racetrack, "_track_length"):
        return

    closed_loop = np.vstack((racetrack.centerline, racetrack.centerline[0]))
    segment_lengths = np.linalg.norm(np.diff(closed_loop, axis=0), axis=1)
    racetrack._segment_lengths = segment_lengths
    racetrack._track_length = float(np.sum(segment_lengths))
    n_points = racetrack.centerline.shape[0]

    if n_points > 0:
        racetrack._avg_segment_length = racetrack._track_length / float(n_points)
        arc_lengths = np.zeros(n_points, dtype=float)
        arc_lengths[1:] = np.cumsum(segment_lengths[:-1])
        racetrack._arc_lengths = arc_lengths

def _lookahead_point(
    centerline: np.ndarray, start_idx: int, distance: float
) -> Tuple[int, np.ndarray]:
    n_points = centerline.shape[0]
    idx = start_idx
    distance = float(max(distance, 0.0))

    if distance == 0.0:
        return idx, centerline[idx]

    travelled = 0.0
    safety_counter = 0

    while travelled < distance and safety_counter < 2 * n_points:
        next_idx = (idx + 1) % n_points
        step = np.linalg.norm(centerline[next_idx] - centerline[idx])
        travelled += step
        idx = next_idx
        safety_counter += 1

        if step < 1e-6:
            break

    return idx, centerline[idx]


def _select_lookahead_distance(speed: float, curvature: float, track_length: float) -> float:
    dynamic = LOOKAHEAD_MIN + LOOKAHEAD_GAIN * max(speed, 0.0)

    if abs(curvature) > HAIRPIN_CURVATURE:
        dynamic *= HAIRPIN_LOOKAHEAD_SCALE

    max_distance = min(LOOKAHEAD_MAX, 0.5 * track_length)
    return float(np.clip(dynamic, LOOKAHEAD_MIN, max_distance))


def _pure_pursuit_delta(
    position: np.ndarray, heading: float, target: np.ndarray, wheelbase: float
) -> float:
    dx, dy = target - position
    sin_h = np.sin(heading)
    cos_h = np.cos(heading)

    x_local = cos_h * dx + sin_h * dy
    y_local = -sin_h * dx + cos_h * dy
    ld2 = x_local * x_local + y_local * y_local

    if ld2 < 1e-6:
        return 0.0

    delta = np.arctan2(wheelbase * 2.0 * y_local, ld2)
    return float(delta)


def _centerline_curvature(centerline: np.ndarray, idx: int) -> float:
    n_points = centerline.shape[0]
    p_prev = centerline[(idx - 1) % n_points]
    p_curr = centerline[idx]
    p_next = centerline[(idx + 1) % n_points]

    a = np.linalg.norm(p_curr - p_prev)
    b = np.linalg.norm(p_next - p_curr)
    c = np.linalg.norm(p_next - p_prev)
    denom = a * b * c

    if denom < 1e-8:
        return 0.0

    area = 0.5 * np.cross(p_curr - p_prev, p_next - p_prev)
    curvature = 4.0 * area / denom
    return float(curvature)


def _speed_from_curvature(curvature: float, parameters: ArrayLike) -> float:
    max_velocity = float(parameters[5])
    abs_curv = abs(curvature)

    if abs_curv < 1e-4:
        target = max_velocity
    else:
        target = np.sqrt(LAT_a_LIMIT / abs_curv)

    target = np.clip(target, MIN_SPEED_TARGET, max_velocity)
    return float(target)


def _filter_speed_command(raw_speed: float, current_speed: float) -> float:
    global _speed_reference

    if _speed_reference is None:
        _speed_reference = max(current_speed, MIN_SPEED_TARGET)

    blended = (1.0 - SPEED_FILTER_ALPHA) * _speed_reference + SPEED_FILTER_ALPHA * raw_speed
    _speed_reference = blended
    return blended


def _pid_step(
    error: float,
    memory: LoopMemory,
    kp: float,
    ki: float,
    kd: float,
    integral_limit: float,
) -> float:
    integral = memory.integral + 0.5 * (error + memory.previous_error) * DT
    integral = float(np.clip(integral, -integral_limit, integral_limit))
    derivative = (error - memory.previous_error) / DT

    memory.integral = integral
    memory.previous_error = error

    return kp * error + ki * integral + kd * derivative


def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack
) -> ArrayLike:
    global hasStarted
    _ensure_track_cache(racetrack)

    state = np.asarray(state, dtype=float)
    position = state[0:2]
    heading = float(state[4])
    speed = float(state[3])
    centerline = racetrack.centerline
    n_points = centerline.shape[0]

    if n_points == 0:
        return np.array([0.0, MIN_CRUISE_SPEED])

    distances = np.linalg.norm(centerline - position, axis=1)
    nearest_idx = int(np.argmin(distances))

    startDist = float(np.linalg.norm(position - centerline[0]))

    if (not hasStarted) and startDist > LAP_START_DISTANCE:
        hasStarted = True

    avg_segment = getattr(
        racetrack,
        "_avg_segment_length",
        racetrack._track_length / float(max(n_points, 1)),
    )
    arc_lengths = getattr(racetrack, "_arc_lengths", None)

    if arc_lengths is not None and arc_lengths.shape[0] == n_points:
        dist_remaining = float(max(racetrack._track_length - arc_lengths[nearest_idx], 0.0))
    else:
        dist_remaining = float(max((n_points - nearest_idx) * avg_segment, 0.0))

    curvature_here = _centerline_curvature(centerline, nearest_idx)
    lookahead = _select_lookahead_distance(speed, curvature_here, racetrack._track_length)

    if hasStarted and dist_remaining < FINISH_APPROACH_DISTANCE:
        finish_scale = 0.4 + 0.6 * (dist_remaining / FINISH_APPROACH_DISTANCE)
        lookahead = max(LOOKAHEAD_FINISH_FLOOR, lookahead * finish_scale)

    target_idx, target_point = _lookahead_point(centerline, nearest_idx, lookahead)

    if hasStarted and (
        dist_remaining < FINISH_TARGET_RADIUS or startDist < FINISH_TARGET_RADIUS
    ):
        target_idx = 0
        target_point = centerline[0]

    wheelbase = float(parameters[0])
    delta_min, delta_max = float(parameters[1]), float(parameters[4])
    desired_delta = _pure_pursuit_delta(position, heading, target_point, wheelbase)
    desired_delta = float(np.clip(desired_delta, delta_min, delta_max))

    targetDist = target_point - position
    angle_to_target = np.arctan2(targetDist[1], targetDist[0])
    angle = angle_to_target - heading
    alpha = (angle + np.pi) % (2.0 * np.pi) - np.pi

    abs_tan_delta = max(abs(np.tan(desired_delta)), 1e-3)
    v_curve = np.sqrt(LAT_a_LIMIT * wheelbase / abs_tan_delta)
    reference = min(v_curve, float(parameters[5]))
    heading_scale = 1.0 / (1.0 + HEADING_PENALTY_GAIN * abs(alpha))
    reference *= heading_scale

    k_target = abs(_centerline_curvature(centerline, target_idx))
    max_weighted_kappa = 0.0

    if n_points > 0:
        for i in range(1, FUTURE_CURVATURE_STEPS + 1):
            future_idx = (nearest_idx + i * FUTURE_CURVATURE_STRIDE) % n_points
            kappa_future = abs(_centerline_curvature(centerline, future_idx))
            weight = 1.0 / (1.0 + CURVATURE_WEIGHT_DECAY * i)
            max_weighted_kappa = max(max_weighted_kappa, kappa_future * weight)

    curvature_limit = _speed_from_curvature(max(abs(curvature_here), k_target, max_weighted_kappa), parameters)
    reference = min(reference, curvature_limit)

    if hasStarted and dist_remaining < FINISH_APPROACH_DISTANCE:
        finish_ratio = dist_remaining / FINISH_APPROACH_DISTANCE
        finish_limit = FINISH_MIN_SPEED + finish_ratio * (float(parameters[5]) - FINISH_MIN_SPEED)
        reference = min(reference, finish_limit)

    v_floor = MIN_SPEED_TARGET if not hasStarted else MIN_CRUISE_SPEED
    if hasStarted and dist_remaining < FINISH_TARGET_RADIUS:
        v_floor = FINISH_MIN_SPEED

    reference = float(np.clip(reference, v_floor, float(parameters[5])))
    desired_speed = _filter_speed_command(reference, speed)
    desired_speed = float(np.clip(desired_speed, v_floor, float(parameters[5])))

    return np.array([desired_delta, desired_speed])


def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:
    assert desired.shape == (2,)

    state = np.asarray(state, dtype=float)
    desired = np.asarray(desired, dtype=float)
    steer_error = float(desired[0] - state[2])
    speed_error = float(desired[1] - state[3])

    rate = _pid_step(
        steer_error,
        _lateral_loop,
        LAT_KP,
        LAT_KI,
        LAT_KD,
        STEER_INTEGRAL_LIMIT,
    )
    a = _pid_step(
        speed_error,
        _longitudinal_loop,
        LONG_KP,
        LONG_KI,
        LONG_KD,
        SPEED_INTEGRAL_LIMIT,
    )

    rate = float(np.clip(rate, parameters[7], parameters[9]))
    a = float(np.clip(a, parameters[8], parameters[10]))

    return np.array([rate, a])
