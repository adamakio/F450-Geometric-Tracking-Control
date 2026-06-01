#!/usr/bin/env python3
"""
plot_controller_report.py

Load the latest (or a specified) ROS 2 bag for a trajectory, compute controller
performance metrics, and save report-ready PNG plots into:

    <plots_root>/<trajectory>/<timestamp>/

Assumptions:
- rosbag2 storage is sqlite3
- messages are standard ROS 2 message types used by the controller
- desired topics and controller topics follow the latest naming you shared

Examples
--------
# Plot latest hover bag
python3 plot_controller_report.py --trajectory hover

# Plot latest circle bag from a custom bags root
python3 plot_controller_report.py \
    --trajectory circle \
    --bags-root ~/Source/Zouhair/bag_files \
    --plots-root ~/Source/Zouhair/report_plots \
    --latest

# Plot a specific bag folder
python3 plot_controller_report.py \
    --trajectory figure8 \
    --bag ~/Source/Zouhair/bag_files/figure8/figure8_20260412_142530
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message

# Message types used
from geometry_msgs.msg import PoseStamped, TwistStamped, AccelStamped, PointStamped, Vector3Stamped
from std_msgs.msg import Float64, Float64MultiArray


# ------------------------------ Configuration ------------------------------

DEFAULT_BAGS_ROOT = Path("~/Source/Zouhair/bag_files").expanduser()
DEFAULT_PLOTS_ROOT = Path("~/Source/Zouhair/report_plots").expanduser()
COMPONENT_COLORS = ["tab:blue", "tab:orange", "tab:green"]

RECORDED_TOPICS = {
    "/uav_0/state/pose",
    "/uav_0/state/twist",
    "/uav_0/state/twist_inertial",
    "/uav_0/state/accel",
    "/uav_0/state/jerk",
    "/uav_0/trajectory/desired/position",
    "/uav_0/trajectory/desired/velocity",
    "/uav_0/trajectory/desired/acceleration",
    "/uav_0/trajectory/desired/jerk",
    "/uav_0/trajectory/desired/snap",
    "/uav_0/trajectory/desired/yaw",
    "/uav_0/trajectory/desired/yaw_rate",
    "/uav_0/trajectory/desired/yaw_acceleration",
    "/uav_0/control/internal/e_R",
    "/uav_0/control/internal/e_omega",
    "/uav_0/control/internal/e_v",
    "/uav_0/control/internal/e_x",
    "/uav_0/control/internal/omega_des",
    "/uav_0/control/internal/alpha_des",
    "/uav_0/control/internal/roll_des",
    "/uav_0/control/internal/pitch_des",
    "/uav_0/control/output/f",
    "/uav_0/control/output/M",
    "/uav_0/control/output/forces",
    "/uav_0/control/output/normalized_throttle",
    "/uav_0/control/rotor0/ref",
    "/uav_0/control/rotor1/ref",
    "/uav_0/control/rotor2/ref",
    "/uav_0/control/rotor3/ref",
}


# ------------------------------ Data classes -------------------------------

@dataclass
class TimeSeries:
    t: np.ndarray  # seconds, relative to bag start or chosen t0
    y: np.ndarray  # shape (N,) or (N, d)

    def empty(self) -> bool:
        return self.t.size == 0


@dataclass
class BagSelection:
    trajectory: str
    bag_dir: Path
    timestamp: str


# ------------------------------ Basic math ---------------------------------

def quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to rotation matrix."""
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("Zero quaternion cannot be converted to a rotation matrix.")
    x, y, z, w = q / n

    return np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),         2 * (x * z + y * w)],
        [2 * (x * y + z * w),         1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
        [2 * (x * z - y * w),         2 * (y * z + x * w),         1 - 2 * (x * x + y * y)],
    ], dtype=float)


def vee(S: np.ndarray) -> np.ndarray:
    """Vee map for a 3x3 skew-symmetric matrix."""
    S = np.asarray(S, dtype=float).reshape(3, 3)
    return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=float)


def so3_attitude_error(R: np.ndarray, R_d: np.ndarray) -> np.ndarray:
    """e_R = 0.5 * vee(R_d^T R - R^T R_d)."""
    return 0.5 * vee(R_d.T @ R - R.T @ R_d)


def attitude_angle_error(R: np.ndarray, R_d: np.ndarray) -> float:
    """Geodesic angle between R and R_d."""
    tr = np.trace(R_d.T @ R)
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cos_theta))


def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError(f"Cannot normalize vector with norm {n:.3e}")
    return v / n


def normalize_with_derivatives(
    p: np.ndarray,
    p_dot: np.ndarray,
    p_ddot: np.ndarray,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    u = p / ||p|| with first and second derivatives.
    """
    s = np.linalg.norm(p)
    if s < eps:
        raise ValueError(f"Cannot normalize vector: norm {s:.3e} below eps={eps:.3e}")

    u = p / s
    P = np.eye(3) - np.outer(u, u)
    s_dot = float(p @ p_dot) / s
    u_dot = (P @ p_dot) / s
    P_dot = -(np.outer(u_dot, u) + np.outer(u, u_dot))
    u_ddot = (P_dot @ p_dot + P @ p_ddot) / s - (s_dot / s) * u_dot
    return u, u_dot, u_ddot


def build_desired_attitude_and_rates(
    a_des: np.ndarray,
    j_des: np.ndarray,
    s_des: np.ndarray,
    yaw_des: float,
    yaw_rate_des: float,
    yaw_acc_des: float,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build R_des, omega_des, alpha_des, and [roll_des, pitch_des, yaw_des]
    from feedforward trajectory terms only.

    This uses:
        A = g e3 + a_des
    instead of the full feedback-shaped A used inside the controller.
    That is the correct offline reconstruction when you only want the
    trajectory-induced desired attitude.
    """
    e3 = np.array([0.0, 0.0, 1.0], dtype=float)

    A = 9.8065 * e3 + a_des
    A_dot = j_des
    A_ddot = s_des

    b3, b3_dot, b3_ddot = normalize_with_derivatives(A, A_dot, A_ddot, eps=eps)

    cy = math.cos(yaw_des)
    sy = math.sin(yaw_des)
    c = np.array([cy, sy, 0.0], dtype=float)
    c_dot = np.array([-sy * yaw_rate_des, cy * yaw_rate_des, 0.0], dtype=float)
    c_ddot = np.array([
        -cy * yaw_rate_des**2 - sy * yaw_acc_des,
        -sy * yaw_rate_des**2 + cy * yaw_acc_des,
        0.0,
    ], dtype=float)

    r = np.cross(b3, c)
    r_dot = np.cross(b3_dot, c) + np.cross(b3, c_dot)
    r_ddot = np.cross(b3_ddot, c) + 2.0 * np.cross(b3_dot, c_dot) + np.cross(b3, c_ddot)

    b2, b2_dot, b2_ddot = normalize_with_derivatives(r, r_dot, r_ddot, eps=eps)

    b1 = np.cross(b2, b3)
    b1_dot = np.cross(b2_dot, b3) + np.cross(b2, b3_dot)
    b1_ddot = np.cross(b2_ddot, b3) + 2.0 * np.cross(b2_dot, b3_dot) + np.cross(b2, b3_ddot)

    R_d = np.column_stack((b1, b2, b3))
    omega_d = np.array([
        b3 @ b2_dot,
        b1 @ b3_dot,
        b2 @ b1_dot,
    ], dtype=float)
    alpha_d = np.array([
        b3_dot @ b2_dot + b3 @ b2_ddot,
        b1_dot @ b3_dot + b1 @ b3_ddot,
        b2_dot @ b1_dot + b2 @ b1_ddot,
    ], dtype=float)

    roll_des = math.atan2(R_d[2, 1], R_d[2, 2])
    pitch_des = math.atan2(-R_d[2, 0], math.hypot(R_d[2, 1], R_d[2, 2]))
    yaw_out = math.atan2(R_d[1, 0], R_d[0, 0])

    return R_d, omega_d, alpha_d, np.array([roll_des, pitch_des, yaw_out], dtype=float)


def rotmat_to_rpy(R: np.ndarray) -> np.ndarray:
    roll = math.atan2(R[2, 1], R[2, 2])
    pitch = math.atan2(-R[2, 0], math.hypot(R[2, 1], R[2, 2]))
    yaw = math.atan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw], dtype=float)


def interp_scalar(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    return np.interp(t_dst, t_src, y_src)


def interp_vector(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    y_src = np.asarray(y_src)
    if y_src.ndim == 1:
        return np.interp(t_dst, t_src, y_src)
    cols = [np.interp(t_dst, t_src, y_src[:, i]) for i in range(y_src.shape[1])]
    return np.column_stack(cols)


def safe_norm_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return np.abs(x)
    return np.linalg.norm(x, axis=1)


# ------------------------------ Bag loading --------------------------------

def find_latest_bag(trajectory: str, bags_root: Path) -> BagSelection:
    traj_dir = bags_root / trajectory
    if not traj_dir.exists():
        raise FileNotFoundError(f"Trajectory folder not found: {traj_dir}")

    candidates = [p for p in traj_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No bag directories found under {traj_dir}")

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    timestamp = latest.name.split("_")[-2] + "_" + latest.name.split("_")[-1] if len(latest.name.split("_")) >= 3 else latest.name
    return BagSelection(trajectory=trajectory, bag_dir=latest, timestamp=timestamp)


def infer_timestamp_from_bag_dir(bag_dir: Path) -> str:
    parts = bag_dir.name.split("_")
    if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
        return f"{parts[-2]}_{parts[-1]}"
    return bag_dir.name


def find_sqlite_db(bag_dir: Path) -> Path:
    dbs = list(bag_dir.glob("*.db3"))
    if not dbs:
        raise FileNotFoundError(f"No .db3 file found in bag directory {bag_dir}")
    if len(dbs) > 1:
        # Usually there is only one. Pick the largest as a sensible fallback.
        return max(dbs, key=lambda p: p.stat().st_size)
    return dbs[0]


def load_bag_topics(bag_dir: Path) -> Dict[str, TimeSeries]:
    db_path = find_sqlite_db(bag_dir)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("SELECT id, name, type FROM topics")
    topic_rows = cur.fetchall()
    topic_map: Dict[int, Tuple[str, str]] = {row[0]: (row[1], row[2]) for row in topic_rows}

    topic_buffers: Dict[str, List[Tuple[float, Any]]] = {}

    cur.execute("SELECT timestamp, topic_id, data FROM messages ORDER BY timestamp ASC")
    rows = cur.fetchall()
    conn.close()

    if not rows:
        raise RuntimeError(f"No messages found in bag database {db_path}")

    t0_ns = rows[0][0]

    for timestamp_ns, topic_id, raw in rows:
        if topic_id not in topic_map:
            continue

        topic_name, topic_type = topic_map[topic_id]
        if topic_name not in RECORDED_TOPICS:
            continue

        msg_cls = get_message(topic_type)
        msg = deserialize_message(raw, msg_cls)
        t = (timestamp_ns - t0_ns) * 1e-9

        value = message_to_numpy(msg)
        topic_buffers.setdefault(topic_name, []).append((t, value))

    out: Dict[str, TimeSeries] = {}
    for topic, samples in topic_buffers.items():
        times = np.array([s[0] for s in samples], dtype=float)
        vals = np.array([s[1] for s in samples], dtype=float)
        out[topic] = TimeSeries(t=times, y=vals)

    return out


def message_to_numpy(msg: Any) -> np.ndarray:
    if isinstance(msg, PoseStamped):
        return np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ], dtype=float)

    if isinstance(msg, TwistStamped):
        return np.array([
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z,
            msg.twist.angular.x,
            msg.twist.angular.y,
            msg.twist.angular.z,
        ], dtype=float)

    if isinstance(msg, AccelStamped):
        return np.array([
            msg.accel.linear.x,
            msg.accel.linear.y,
            msg.accel.linear.z,
            msg.accel.angular.x,
            msg.accel.angular.y,
            msg.accel.angular.z,
        ], dtype=float)

    if isinstance(msg, PointStamped):
        return np.array([msg.point.x, msg.point.y, msg.point.z], dtype=float)

    if isinstance(msg, Vector3Stamped):
        return np.array([msg.vector.x, msg.vector.y, msg.vector.z], dtype=float)

    if isinstance(msg, Float64):
        return np.array([msg.data], dtype=float)

    if isinstance(msg, Float64MultiArray):
        return np.array(list(msg.data), dtype=float)

    raise TypeError(f"Unsupported message type: {type(msg).__name__}")


# ------------------------------ Derived data -------------------------------

def require_topic(topics: Dict[str, TimeSeries], name: str) -> TimeSeries:
    if name not in topics:
        raise KeyError(f"Missing required topic in bag: {name}")
    return topics[name]


def pick_common_timebase(topics: Dict[str, TimeSeries]) -> np.ndarray:
    """
    Use state/pose timestamps as the master time base. This is usually the most
    natural reference for controller performance plots.
    """
    pose = require_topic(topics, "/uav_0/state/pose")
    return pose.t.copy()


def build_dataset(topics: Dict[str, TimeSeries]) -> Dict[str, np.ndarray]:
    t = pick_common_timebase(topics)

    pose = require_topic(topics, "/uav_0/state/pose")
    twist = require_topic(topics, "/uav_0/state/twist")
    twist_i = require_topic(topics, "/uav_0/state/twist_inertial")
    accel = require_topic(topics, "/uav_0/state/accel")
    jerk = require_topic(topics, "/uav_0/state/jerk")

    pos_d = require_topic(topics, "/uav_0/trajectory/desired/position")
    vel_d = require_topic(topics, "/uav_0/trajectory/desired/velocity")
    acc_d = require_topic(topics, "/uav_0/trajectory/desired/acceleration")
    jerk_d = require_topic(topics, "/uav_0/trajectory/desired/jerk")
    snap_d = require_topic(topics, "/uav_0/trajectory/desired/snap")
    yaw_d = require_topic(topics, "/uav_0/trajectory/desired/yaw")
    yaw_rate_d = require_topic(topics, "/uav_0/trajectory/desired/yaw_rate")
    yaw_acc_d = require_topic(topics, "/uav_0/trajectory/desired/yaw_acceleration")

    x = interp_vector(pose.t, pose.y[:, 0:3], t)
    q = interp_vector(pose.t, pose.y[:, 3:7], t)
    omega = interp_vector(twist.t, twist.y[:, 3:6], t)
    v = interp_vector(twist_i.t, twist_i.y[:, 0:3], t)
    a = interp_vector(accel.t, accel.y[:, 0:3], t)
    alpha = interp_vector(accel.t, accel.y[:, 3:6], t)
    j = interp_vector(jerk.t, jerk.y[:, 0:3], t)

    x_des = interp_vector(pos_d.t, pos_d.y[:, 0:3], t)
    v_des = interp_vector(vel_d.t, vel_d.y[:, 0:3], t)
    a_des = interp_vector(acc_d.t, acc_d.y[:, 0:3], t)
    j_des = interp_vector(jerk_d.t, jerk_d.y[:, 0:3], t)
    s_des = interp_vector(snap_d.t, snap_d.y[:, 0:3], t)
    yaw_des = interp_scalar(yaw_d.t, yaw_d.y[:, 0], t)
    yaw_rate_des = interp_scalar(yaw_rate_d.t, yaw_rate_d.y[:, 0], t)
    yaw_acc_des = interp_scalar(yaw_acc_d.t, yaw_acc_d.y[:, 0], t)

    # Optional direct controller topics if present
    omega_des_topic = topics.get("/uav_0/control/internal/omega_des")
    alpha_des_topic = topics.get("/uav_0/control/internal/alpha_des")
    e_R_topic = topics.get("/uav_0/control/internal/e_R")
    e_omega_topic = topics.get("/uav_0/control/internal/e_omega")
    roll_des_topic = topics.get("/uav_0/control/internal/roll_des")
    pitch_des_topic = topics.get("/uav_0/control/internal/pitch_des")
    f_topic = topics.get("/uav_0/control/output/f")
    M_topic = topics.get("/uav_0/control/output/M")
    throttle_topic = topics.get("/uav_0/control/output/normalized_throttle")

    omega_des_meas = interp_vector(omega_des_topic.t, omega_des_topic.y, t) if omega_des_topic else None
    alpha_des_meas = interp_vector(alpha_des_topic.t, alpha_des_topic.y, t) if alpha_des_topic else None
    e_R_meas = interp_vector(e_R_topic.t, e_R_topic.y, t) if e_R_topic else None
    e_omega_meas = interp_vector(e_omega_topic.t, e_omega_topic.y, t) if e_omega_topic else None
    roll_des_meas = interp_scalar(roll_des_topic.t, roll_des_topic.y[:, 0], t) if roll_des_topic else None
    pitch_des_meas = interp_scalar(pitch_des_topic.t, pitch_des_topic.y[:, 0], t) if pitch_des_topic else None
    f = interp_scalar(f_topic.t, f_topic.y[:, 0], t) if f_topic else None
    M = interp_vector(M_topic.t, M_topic.y, t) if M_topic else None
    throttle = interp_vector(throttle_topic.t, throttle_topic.y, t) if throttle_topic else None

    N = t.size
    R = np.zeros((N, 3, 3), dtype=float)
    R_des = np.zeros((N, 3, 3), dtype=float)
    rpy = np.zeros((N, 3), dtype=float)
    rpy_des = np.zeros((N, 3), dtype=float)
    omega_des_calc = np.zeros((N, 3), dtype=float)
    alpha_des_calc = np.zeros((N, 3), dtype=float)
    e_R_calc = np.zeros((N, 3), dtype=float)
    e_omega_calc = np.zeros((N, 3), dtype=float)
    theta_e = np.zeros(N, dtype=float)

    for i in range(N):
        R[i] = quat_xyzw_to_rotmat(q[i])
        rpy[i] = rotmat_to_rpy(R[i])

        try:
            R_d_i, omega_d_i, alpha_d_i, rpy_d_i = build_desired_attitude_and_rates(
                a_des=a_des[i],
                j_des=j_des[i],
                s_des=s_des[i],
                yaw_des=float(yaw_des[i]),
                yaw_rate_des=float(yaw_rate_des[i]),
                yaw_acc_des=float(yaw_acc_des[i]),
            )
        except ValueError:
            # In singular or degenerate cases, carry forward the previous value if possible
            if i == 0:
                R_d_i = np.eye(3)
                omega_d_i = np.zeros(3)
                alpha_d_i = np.zeros(3)
                rpy_d_i = np.zeros(3)
            else:
                R_d_i = R_des[i - 1]
                omega_d_i = omega_des_calc[i - 1]
                alpha_d_i = alpha_des_calc[i - 1]
                rpy_d_i = rpy_des[i - 1]

        R_des[i] = R_d_i
        omega_des_calc[i] = omega_d_i
        alpha_des_calc[i] = alpha_d_i
        rpy_des[i] = rpy_d_i

        e_R_calc[i] = so3_attitude_error(R[i], R_d_i)
        e_omega_calc[i] = omega[i] - R[i].T @ R_d_i @ omega_d_i
        theta_e[i] = attitude_angle_error(R[i], R_d_i)

    # Use measured controller topics when available, else computed values
    omega_des_use = omega_des_meas if omega_des_meas is not None else omega_des_calc
    alpha_des_use = alpha_des_meas if alpha_des_meas is not None else alpha_des_calc
    e_R_use = e_R_meas if e_R_meas is not None else e_R_calc
    e_omega_use = e_omega_meas if e_omega_meas is not None else e_omega_calc

    if roll_des_meas is not None:
        rpy_des[:, 0] = roll_des_meas
    if pitch_des_meas is not None:
        rpy_des[:, 1] = pitch_des_meas
    # Always use yaw_des from trajectory
    rpy_des[:, 2] = yaw_des

    e_x = x - x_des
    e_v = v - v_des

    return {
        "t": t,
        "x": x,
        "q": q,
        "omega": omega,
        "v": v,
        "a": a,
        "alpha": alpha,
        "j": j,
        "x_des": x_des,
        "v_des": v_des,
        "a_des": a_des,
        "j_des": j_des,
        "s_des": s_des,
        "yaw_des": yaw_des,
        "yaw_rate_des": yaw_rate_des,
        "yaw_acc_des": yaw_acc_des,
        "R": R,
        "R_des": R_des,
        "rpy": rpy,
        "rpy_des": rpy_des,
        "omega_des": omega_des_use,
        "alpha_des": alpha_des_use,
        "e_x": e_x,
        "e_v": e_v,
        "e_R": e_R_use,
        "e_omega": e_omega_use,
        "theta_e": theta_e,
        "f": f,
        "M": M,
        "throttle": throttle,
    }


# ------------------------------ Metrics ------------------------------------

def compute_hover_metrics(t: np.ndarray, y: np.ndarray, y_des: np.ndarray) -> Dict[str, float]:
    """
    Basic step-response metrics on a scalar channel.
    Assumes final desired value is the step target.
    """
    y = np.asarray(y, dtype=float)
    y_des = np.asarray(y_des, dtype=float)
    target = float(np.median(y_des[-max(10, len(y_des) // 10):]))
    y0 = float(np.median(y[:max(10, len(y) // 20)]))
    step = target - y0

    metrics: Dict[str, float] = {
        "target": target,
        "initial": y0,
        "step_size": step,
        "steady_state_error": float(np.median(y[-max(10, len(y) // 10):]) - target),
        "rms_error": float(np.sqrt(np.mean((y - y_des) ** 2))),
        "max_abs_error": float(np.max(np.abs(y - y_des))),
    }

    if abs(step) < 1e-9:
        return metrics

    y_norm = (y - y0) / step
    lo, hi = (0.1, 0.9) if step > 0 else (0.9, 0.1)

    def first_cross(level: float) -> Optional[float]:
        idx = np.where(y_norm >= level)[0] if step > 0 else np.where(y_norm <= level)[0]
        return float(t[idx[0]]) if idx.size else None

    t10 = first_cross(lo)
    t90 = first_cross(hi)
    if t10 is not None and t90 is not None:
        metrics["rise_time_10_90"] = t90 - t10

    peak = float(np.max(y)) if step > 0 else float(np.min(y))
    overshoot = (peak - target) / abs(step) * 100.0 if step > 0 else (target - peak) / abs(step) * 100.0
    metrics["overshoot_percent"] = max(0.0, overshoot)

    tol = 0.02 * abs(step)
    settled_idx = np.where(np.abs(y - target) <= tol)[0]
    if settled_idx.size:
        # Settling time is first time after which it remains in bounds
        for idx in settled_idx:
            if np.all(np.abs(y[idx:] - target) <= tol):
                metrics["settling_time_2pct"] = float(t[idx])
                break

    return metrics


def compute_general_metrics(ds: Dict[str, np.ndarray]) -> Dict[str, float]:
    e_x_norm = safe_norm_rows(ds["e_x"])
    e_R_norm = safe_norm_rows(ds["e_R"])
    theta_deg = np.rad2deg(ds["theta_e"])

    metrics: Dict[str, float] = {
        "rms_position_error_m": float(np.sqrt(np.mean(e_x_norm**2))),
        "max_position_error_m": float(np.max(e_x_norm)),
        "mean_position_error_m": float(np.mean(e_x_norm)),
        "rms_attitude_error_deg": float(np.sqrt(np.mean(theta_deg**2))),
        "max_attitude_error_deg": float(np.max(theta_deg)),
    }

    if ds["throttle"] is not None:
        thr = ds["throttle"]
        sat = np.logical_or(thr <= 0.0, thr >= 1.0)
        metrics["throttle_saturation_fraction"] = float(np.mean(sat))
        metrics["throttle_max"] = float(np.max(thr))
        metrics["throttle_min"] = float(np.min(thr))

    return metrics


# ------------------------------ Plot helpers -------------------------------

def setup_matplotlib() -> None:
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.8,
        "grid.alpha": 0.25,
        "axes.grid": True,
        "axes.formatter.use_mathtext": True,
    })


def savefig(path: Path, tight: bool = True) -> None:
    fig = plt.gcf()
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_component_tracking(
    t: np.ndarray,
    actual: np.ndarray,
    desired: np.ndarray,
    ylabel: str,
    title: str,
    out: Path,
    labels: Sequence[str],
) -> None:
    plt.figure(figsize=(7.0, 4.3))
    for i, lbl in enumerate(labels):
        color = COMPONENT_COLORS[i % len(COMPONENT_COLORS)]
        plt.plot(t, actual[:, i], color=color, label=rf"${lbl}$")
    for i, lbl in enumerate(labels):
        color = COMPONENT_COLORS[i % len(COMPONENT_COLORS)]
        plt.plot(t, desired[:, i], "--", color=color, label=rf"${{{lbl}}}_{{\mathrm{{des}}}}$")
    plt.xlabel(r"Time [s]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(ncol=2)
    savefig(out)


def plot_xyz_tracking(t: np.ndarray, x: np.ndarray, x_des: np.ndarray, out: Path) -> None:
    plot_component_tracking(t, x, x_des, r"Position [m]", r"Position Tracking", out, [r"x", r"y", r"z"])


def plot_xy_path(x: np.ndarray, x_des: np.ndarray, out: Path) -> None:
    plt.figure(figsize=(6.0, 6.0))
    plt.plot(x[:, 0], x[:, 1], label=r"Actual")
    plt.plot(x_des[:, 0], x_des[:, 1], "--", label=r"Desired")
    plt.xlabel(r"$x$ [m]")
    plt.ylabel(r"$y$ [m]")
    plt.title(r"XY Path")
    plt.axis("equal")
    plt.legend()
    savefig(out)


def plot_xyz_path_3d(x: np.ndarray, x_des: np.ndarray, out: Path) -> None:
    fig = plt.figure(figsize=(7.0, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x[:, 0], x[:, 1], x[:, 2], label=r"Actual", linewidth=2.0)
    ax.plot(x_des[:, 0], x_des[:, 1], x_des[:, 2], "--", label=r"Desired", linewidth=1.8)
    ax.set_xlabel(r"$x$ [m]", labelpad=10)
    ax.set_ylabel(r"$y$ [m]", labelpad=10)
    ax.set_zlabel(r"$z$ [m]", labelpad=10)
    ax.set_title(r"3D Trajectory")
    ax.legend()
    ax.view_init(elev=24, azim=-58)

    pts = np.vstack((x, x_des))
    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.55 * max(np.max(maxs - mins), 1e-3)

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.92)
    savefig(out, tight=False)


def plot_vector_components(t: np.ndarray, vec: np.ndarray, ylabel: str, title: str, out: Path, labels: Sequence[str]) -> None:
    plt.figure(figsize=(7.0, 4.0))
    for i, lbl in enumerate(labels):
        plt.plot(t, vec[:, i], color=COMPONENT_COLORS[i % len(COMPONENT_COLORS)], label=rf"${lbl}$")
    plt.xlabel(r"Time [s]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    savefig(out)


def plot_vector_components_with_norm(
    t: np.ndarray,
    vec: np.ndarray,
    ylabel: str,
    title: str,
    out: Path,
    labels: Sequence[str],
    norm_label: str,
) -> None:
    plt.figure(figsize=(7.0, 4.2))
    for i, lbl in enumerate(labels):
        plt.plot(t, vec[:, i], color=COMPONENT_COLORS[i % len(COMPONENT_COLORS)], label=rf"${lbl}$")
    plt.plot(t, safe_norm_rows(vec), "k--", label=norm_label)
    plt.xlabel(r"Time [s]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    savefig(out)


def plot_rpy_tracking(t: np.ndarray, rpy: np.ndarray, rpy_des: np.ndarray, out: Path) -> None:
    plot_component_tracking(
        t,
        np.rad2deg(rpy),
        np.rad2deg(rpy_des),
        r"Angle [deg]",
        r"RPY Tracking",
        out,
        [r"\phi", r"\theta", r"\psi"],
    )


def plot_scalar_series(t: np.ndarray, y: np.ndarray, ylabel: str, title: str, out: Path) -> None:
    plt.figure(figsize=(7.0, 4.0))
    plt.plot(t, y)
    plt.xlabel(r"Time [s]")
    plt.ylabel(ylabel)
    plt.title(title)
    savefig(out)


def plot_moments(t: np.ndarray, M: np.ndarray, out: Path) -> None:
    plot_vector_components(t, M, "Moment [N·m]", "Control moments", out, labels=["Mx", "My", "Mz"])


def plot_rotors(t: np.ndarray, throttle: np.ndarray, out: Path) -> None:
    labels = [f"rotor{i}" for i in range(throttle.shape[1])]
    plt.figure(figsize=(7.0, 4.0))
    for i, lbl in enumerate(labels):
        plt.plot(t, throttle[:, i], label=lbl)
    plt.xlabel(r"Time [s]")
    plt.ylabel(r"Normalized throttle [-]")
    plt.title(r"Rotor Commands")
    plt.legend(ncol=2)
    savefig(out)


def write_metrics_txt(path: Path, sections: Dict[str, Dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for section_name, metrics in sections.items():
            f.write(f"[{section_name}]\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value:.6f}\n")
            f.write("\n")


# ------------------------------ Orchestration ------------------------------

def ensure_output_dir(plots_root: Path, trajectory: str, timestamp: str) -> Path:
    outdir = plots_root / trajectory / timestamp
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def make_plots(trajectory: str, ds: Dict[str, np.ndarray], outdir: Path) -> None:
    t = ds["t"]

    # Common set
    plot_xyz_tracking(t, ds["x"], ds["x_des"], outdir / "tracking_xyz.png")
    plot_xy_path(ds["x"], ds["x_des"], outdir / "path_xy.png")
    plot_xyz_path_3d(ds["x"], ds["x_des"], outdir / "path_xyz_3d.png")
    plot_vector_components_with_norm(
        t,
        ds["e_x"],
        r"Position error [m]",
        r"Position Error",
        outdir / "error_position.png",
        [r"e_x", r"e_y", r"e_z"],
        r"$\|e_x\|$",
    )
    plot_component_tracking(
        t,
        ds["v"],
        ds["v_des"],
        r"Velocity [m/s]",
        r"Velocity Tracking",
        outdir / "tracking_velocity.png",
        [r"v_x", r"v_y", r"v_z"],
    )
    plot_vector_components_with_norm(
        t,
        ds["e_v"],
        r"Velocity error [m/s]",
        r"Velocity Error",
        outdir / "error_velocity.png",
        [r"e_{v_x}", r"e_{v_y}", r"e_{v_z}"],
        r"$\|e_v\|$",
    )
    plot_rpy_tracking(t, ds["rpy"], ds["rpy_des"], outdir / "tracking_rpy.png")
    plot_vector_components_with_norm(
        t,
        ds["e_R"],
        r"Attitude error [-]",
        r"Attitude Error $e_R$",
        outdir / "error_attitude.png",
        [r"e_{R_x}", r"e_{R_y}", r"e_{R_z}"],
        r"$\|e_R\|$",
    )
    plot_vector_components_with_norm(
        t,
        ds["e_omega"],
        r"Angular-rate error [rad/s]",
        r"Angular-Rate Error $e_\omega$",
        outdir / "error_omega.png",
        [r"e_{\omega_x}", r"e_{\omega_y}", r"e_{\omega_z}"],
        r"$\|e_\omega\|$",
    )

    if ds["omega_des"] is not None:
        plot_component_tracking(
            t,
            ds["omega"],
            ds["omega_des"],
            r"Angular rate [rad/s]",
            r"Angular Velocity Tracking",
            outdir / "omega_tracking.png",
            [r"\omega_x", r"\omega_y", r"\omega_z"],
        )

    if ds["alpha_des"] is not None and ds["alpha"] is not None:
        plot_component_tracking(
            t,
            ds["alpha"],
            ds["alpha_des"],
            r"Angular acceleration [rad/s$^2$]",
            r"Angular Acceleration Tracking",
            outdir / "alpha_tracking.png",
            [r"\alpha_x", r"\alpha_y", r"\alpha_z"],
        )

    if ds["f"] is not None:
        plot_scalar_series(t, ds["f"], r"Thrust [N]", r"Total Thrust", outdir / "control_thrust.png")

    if ds["M"] is not None:
        plot_moments(t, ds["M"], outdir / "control_moments.png")

    if ds["throttle"] is not None:
        plot_rotors(t, ds["throttle"], outdir / "control_rotors.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report-ready controller plots from a ROS 2 bag.")
    parser.add_argument("--trajectory", required=True, help="Trajectory name, e.g. hover, circle, figure8")
    parser.add_argument("--bags-root", type=Path, default=DEFAULT_BAGS_ROOT, help="Root folder containing bag_files/<trajectory>/...")
    parser.add_argument("--plots-root", type=Path, default=DEFAULT_PLOTS_ROOT, help="Root folder to save plots into")
    parser.add_argument("--bag", type=Path, default=None, help="Specific bag directory to load")
    parser.add_argument("--latest", action="store_true", help="Use the latest bag for the trajectory (default behavior if --bag is not given)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_matplotlib()

    trajectory = args.trajectory
    bags_root = args.bags_root.expanduser()
    plots_root = args.plots_root.expanduser()

    if args.bag is not None:
        bag_dir = args.bag.expanduser()
        if not bag_dir.exists():
            raise FileNotFoundError(f"Specified bag directory does not exist: {bag_dir}")
        timestamp = infer_timestamp_from_bag_dir(bag_dir)
        selection = BagSelection(trajectory=trajectory, bag_dir=bag_dir, timestamp=timestamp)
    else:
        selection = find_latest_bag(trajectory, bags_root)

    print(f"Loading bag: {selection.bag_dir}")
    topics = load_bag_topics(selection.bag_dir)
    ds = build_dataset(topics)

    outdir = ensure_output_dir(plots_root, selection.trajectory, selection.timestamp)
    print(f"Saving plots to: {outdir}")

    make_plots(selection.trajectory, ds, outdir)

    general_metrics = compute_general_metrics(ds)
    metrics_sections: Dict[str, Dict[str, float]] = {"general": general_metrics}

    if selection.trajectory.lower() == "hover":
        hover_metrics = compute_hover_metrics(ds["t"], ds["x"][:, 2], ds["x_des"][:, 2])
        metrics_sections["hover_z"] = hover_metrics

    write_metrics_txt(outdir / "metrics_summary.txt", metrics_sections)

    print("Done.")
    print(f"Plots and metrics saved under: {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
