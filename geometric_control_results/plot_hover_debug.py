#!/usr/bin/env python3
"""
Plot geometric controller debug data from a ROS2 bag file.

Usage:
    python3 plot_hover_debug.py <path_to_bag_dir>

Example:
    python3 plot_hover_debug.py ~/Workspaces/fsc_autopilot_ws/bag_files/hover_debug
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
except ImportError:
    print("ERROR: rosbags not installed. Run:\n  pip install rosbags")
    sys.exit(1)


# ── Bag reader ─────────────────────────────────────────────────────────────────

def read_bag(bag_path: str) -> dict:
    """
    Read all relevant topics from a ROS2 bag file.
    Returns a dict of {topic: {"t": np.array, "data": np.array}}.
    """
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    # Maps topic → expected field accessor (lambda msg → np.array)
    topic_parsers = {
        "/uav_0/debug/state/x":                   lambda m: np.array([m.vector.x, m.vector.y, m.vector.z]),
        "/uav_0/debug/state/v":                   lambda m: np.array([m.vector.x, m.vector.y, m.vector.z]),
        "/uav_0/debug/state/omega":               lambda m: np.array([m.vector.x, m.vector.y, m.vector.z]),
        "/uav_0/debug/state/q":                   lambda m: np.array([m.quaternion.x, m.quaternion.y,
                                                                        m.quaternion.z, m.quaternion.w]),
        "/uav_0/debug/controller/e_x":            lambda m: np.array([m.vector.x, m.vector.y, m.vector.z]),
        "/uav_0/debug/controller/e_R":            lambda m: np.array([m.vector.x, m.vector.y, m.vector.z]),
        "/uav_0/debug/controller/e_omega":        lambda m: np.array([m.vector.x, m.vector.y, m.vector.z]),
        "/uav_0/debug/controller/b3_des":         lambda m: np.array([m.vector.x, m.vector.y, m.vector.z]),
        "/uav_0/debug/output/f":                  lambda m: np.array([m.data]),
        "/uav_0/debug/output/normalized_throttle":lambda m: np.array(m.data),
    }

    data = {topic: {"t": [], "data": []} for topic in topic_parsers}

    with Reader(bag_path) as reader:
        connections = {c.topic: c for c in reader.connections}
        for topic in topic_parsers:
            if topic not in connections:
                print(f"  [WARN] Topic not found in bag: {topic}")

        for connection, timestamp, rawdata in reader.messages():
            topic = connection.topic
            if topic not in topic_parsers:
                continue
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            t_sec = timestamp * 1e-9  # nanoseconds → seconds
            data[topic]["t"].append(t_sec)
            data[topic]["data"].append(topic_parsers[topic](msg))

    # Convert to numpy arrays
    for topic in data:
        if len(data[topic]["t"]) == 0:
            data[topic]["t"] = np.array([])
            data[topic]["data"] = np.array([])
            continue
        data[topic]["t"] = np.array(data[topic]["t"])
        data[topic]["data"] = np.array(data[topic]["data"])

    # Detect motion start: first index where position moves more than 1mm from t=0 value
    t_start = 0.0
    pos_topic = "/uav_0/debug/state/x"
    if len(data[pos_topic]["t"]) > 0:
        pos = data[pos_topic]["data"]
        delta = np.linalg.norm(pos - pos[0], axis=1)
        moving = np.where(delta > 1e-3)[0]
        if len(moving) > 0:
            t_start = data[pos_topic]["t"][moving[0]]
            print(f"  Motion detected at t = {t_start:.3f} s — trimming flat prefix")

    # Trim all topics to t >= t_start and normalise time to 0
    for topic in data:
        if len(data[topic]["t"]) == 0:
            continue
        t = data[topic]["t"]
        mask = t >= t_start
        data[topic]["t"] = t[mask] - t_start
        data[topic]["data"] = data[topic]["data"][mask]

    return data


# ── Helpers ────────────────────────────────────────────────────────────────────

def get(data, topic):
    """Return (t, arr) or (None, None) if missing/empty."""
    d = data.get(topic, {})
    t = d.get("t", np.array([]))
    arr = d.get("data", np.array([]))
    if len(t) == 0:
        return None, None
    return t, arr


def quat_to_euler_deg(q_arr):
    """Convert (N,4) [x,y,z,w] quaternions to (N,3) Euler angles ZYX in degrees."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(q_arr).as_euler('ZYX', degrees=True)[:, ::-1]  # roll, pitch, yaw


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_position(data, fig, gs_row, gs_col):
    ax = fig.add_subplot(gs_row, gs_col)
    t, x = get(data, "/uav_0/debug/state/x")
    t2, ex = get(data, "/uav_0/debug/controller/e_x")
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    labels = ['x', 'y', 'z']
    if t is not None:
        for i in range(3):
            ax.plot(t, x[:, i], color=colors[i], label=labels[i])
        ax.axhline(0.0, color='tab:blue',   linestyle='--', alpha=0.4)
        ax.axhline(0.0, color='tab:orange', linestyle='--', alpha=0.4)
        ax.axhline(1.0, color='tab:green',  linestyle='--', alpha=0.4, label='z_des=1')
    ax.set_title('Position (m)')
    ax.set_xlabel('t (s)')
    ax.legend(fontsize=7)
    ax.grid(True)
    return ax


def plot_position_error(data, fig, gs_slot):
    ax = fig.add_subplot(gs_slot)
    t, ex = get(data, "/uav_0/debug/controller/e_x")
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    if t is not None:
        for i, lb in enumerate(['ex', 'ey', 'ez']):
            ax.plot(t, ex[:, i], color=colors[i], label=lb)
        ax.plot(t, np.linalg.norm(ex, axis=1), 'k--', linewidth=1.5, label='‖ex‖')
    ax.set_title('Position error (m)')
    ax.set_xlabel('t (s)')
    ax.legend(fontsize=7)
    ax.grid(True)
    return ax


def plot_attitude(data, fig, gs_slot):
    ax = fig.add_subplot(gs_slot)
    t, q = get(data, "/uav_0/debug/state/q")
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    if t is not None:
        euler = quat_to_euler_deg(q)
        for i, lb in enumerate(['roll', 'pitch', 'yaw']):
            ax.plot(t, euler[:, i], color=colors[i], label=lb)
    ax.set_title('Attitude — Euler angles (deg)')
    ax.set_xlabel('t (s)')
    ax.legend(fontsize=7)
    ax.grid(True)
    return ax


def plot_attitude_error(data, fig, gs_slot):
    ax = fig.add_subplot(gs_slot)
    t, eR = get(data, "/uav_0/debug/controller/e_R")
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    if t is not None:
        for i, lb in enumerate(['eR_x', 'eR_y', 'eR_z']):
            ax.plot(t, eR[:, i], color=colors[i], label=lb)
        ax.plot(t, np.linalg.norm(eR, axis=1), 'k--', linewidth=1.5, label='‖eR‖')
    ax.set_title('Attitude error e_R (rad)')
    ax.set_xlabel('t (s)')
    ax.legend(fontsize=7)
    ax.grid(True)
    return ax


def plot_omega_error(data, fig, gs_slot):
    ax = fig.add_subplot(gs_slot)
    t, ew = get(data, "/uav_0/debug/controller/e_omega")
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    if t is not None:
        for i, lb in enumerate(['eΩ_x', 'eΩ_y', 'eΩ_z']):
            ax.plot(t, ew[:, i], color=colors[i], label=lb)
    ax.set_title('Angular rate error e_Ω (rad/s)')
    ax.set_xlabel('t (s)')
    ax.legend(fontsize=7)
    ax.grid(True)
    return ax


def plot_b3_des(data, fig, gs_slot):
    ax = fig.add_subplot(gs_slot)
    t, b3 = get(data, "/uav_0/debug/controller/b3_des")
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    if t is not None:
        for i, lb in enumerate(['b3x', 'b3y', 'b3z']):
            ax.plot(t, b3[:, i], color=colors[i], label=lb)
        ax.axhline(1.0, color='tab:green', linestyle='--', alpha=0.5, label='ideal b3z=1')
    ax.set_title('Desired b3 axis (should be ≈[0,0,1] at hover)')
    ax.set_xlabel('t (s)')
    ax.legend(fontsize=7)
    ax.grid(True)
    return ax


def plot_thrust(data, fig, gs_slot):
    ax = fig.add_subplot(gs_slot)
    t, f = get(data, "/uav_0/debug/output/f")
    mg = 1.806 * 9.8065
    if t is not None:
        ax.plot(t, f[:, 0], color='tab:brown', label='f')
        ax.axhline(mg, color='gray', linestyle='--', label=f'mg={mg:.1f} N')
        ax.axhline(0,  color='k',    linestyle='--', alpha=0.3)
    ax.set_title('Total thrust f (N)')
    ax.set_xlabel('t (s)')
    ax.legend(fontsize=7)
    ax.grid(True)
    return ax


def plot_throttle(data, fig, gs_slot):
    ax = fig.add_subplot(gs_slot)
    t, thr = get(data, "/uav_0/debug/output/normalized_throttle")
    rotor_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    if t is not None:
        n_rotors = thr.shape[1] if thr.ndim > 1 else 1
        for i in range(n_rotors):
            col = thr[:, i] if thr.ndim > 1 else thr
            ax.plot(t, col, color=rotor_colors[i], label=f'rotor{i}')
        ax.axhline(0.0, color='k', linestyle='--', alpha=0.4, label='0 (min)')
        ax.axhline(1.0, color='k', linestyle=':',  alpha=0.4, label='1 (max)')
    ax.set_title('Normalized throttle per rotor')
    ax.set_xlabel('t (s)')
    ax.legend(fontsize=7)
    ax.grid(True)
    return ax


def plot_omega(data, fig, gs_slot):
    ax = fig.add_subplot(gs_slot)
    t, w = get(data, "/uav_0/debug/state/omega")
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    if t is not None:
        for i, lb in enumerate(['ωx', 'ωy', 'ωz']):
            ax.plot(t, w[:, i], color=colors[i], label=lb)
    ax.set_title('Body angular velocity ω (rad/s)')
    ax.set_xlabel('t (s)')
    ax.legend(fontsize=7)
    ax.grid(True)
    return ax


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    bag_path = sys.argv[1]
    bag_name = bag_path.split("/")[-1]
    print(f"Reading bag: {bag_path}")
    data = read_bag(bag_path)

    # Print quick summary
    print("\n── Data summary ──────────────────────────────────────")
    for topic, d in data.items():
        n = len(d["t"])
        if n > 0:
            dur = d["t"][-1] - d["t"][0]
            print(f"  {topic:55s}  {n:5d} msgs  {dur:.2f} s")
        else:
            print(f"  {topic:55s}  [EMPTY]")

    # ── Figure 1: Position & attitude overview ─────────────────────────────────
    fig1, axes1 = plt.subplots(3, 3, figsize=(17, 11))
    fig1.suptitle('Geometric Controller — Hover Debug', fontsize=13, fontweight='bold')

    slot = lambda r, c: axes1[r, c]

    # Row 0
    t, x  = get(data, "/uav_0/debug/state/x")
    ax = axes1[0, 0]
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    if t is not None:
        for i, lb in enumerate(['x', 'y', 'z']):
            ax.plot(t, x[:, i], color=colors[i], label=lb)
        ax.axhline(1.0, color='tab:green', linestyle='--', alpha=0.5, label='z_des=1')
    ax.set_title('Position (m)'); ax.legend(fontsize=7); ax.grid(True); ax.set_xlabel('t (s)')

    t2, ex = get(data, "/uav_0/debug/controller/e_x")
    ax = axes1[0, 1]
    if t2 is not None:
        for i, lb in enumerate(['ex', 'ey', 'ez']):
            ax.plot(t2, ex[:, i], color=colors[i], label=lb)
        ax.plot(t2, np.linalg.norm(ex, axis=1), 'k--', linewidth=1.5, label='‖ex‖')
    ax.set_title('Position error (m)'); ax.legend(fontsize=7); ax.grid(True); ax.set_xlabel('t (s)')

    t3, q = get(data, "/uav_0/debug/state/q")
    ax = axes1[0, 2]
    if t3 is not None:
        euler = quat_to_euler_deg(q)
        for i, lb in enumerate(['roll', 'pitch', 'yaw']):
            ax.plot(t3, euler[:, i], color=colors[i], label=lb)
    ax.set_title('Attitude — Euler (deg)'); ax.legend(fontsize=7); ax.grid(True); ax.set_xlabel('t (s)')

    # Row 1
    t4, eR = get(data, "/uav_0/debug/controller/e_R")
    ax = axes1[1, 0]
    if t4 is not None:
        for i, lb in enumerate(['eR_x', 'eR_y', 'eR_z']):
            ax.plot(t4, eR[:, i], color=colors[i], label=lb)
        ax.plot(t4, np.linalg.norm(eR, axis=1), 'k--', linewidth=1.5, label='‖eR‖')
    ax.set_title('Attitude error e_R (rad)'); ax.legend(fontsize=7); ax.grid(True); ax.set_xlabel('t (s)')

    t5, ew = get(data, "/uav_0/debug/controller/e_omega")
    ax = axes1[1, 1]
    if t5 is not None:
        for i, lb in enumerate(['eΩ_x', 'eΩ_y', 'eΩ_z']):
            ax.plot(t5, ew[:, i], color=colors[i], label=lb)
    ax.set_title('Angular rate error e_Ω (rad/s)'); ax.legend(fontsize=7); ax.grid(True); ax.set_xlabel('t (s)')

    t6, b3 = get(data, "/uav_0/debug/controller/b3_des")
    ax = axes1[1, 2]
    if t6 is not None:
        for i, lb in enumerate(['b3x', 'b3y', 'b3z']):
            ax.plot(t6, b3[:, i], color=colors[i], label=lb)
        ax.axhline(1.0, color='tab:green', linestyle='--', alpha=0.5, label='ideal=1')
    ax.set_title('Desired b3 (should be ≈[0,0,1])'); ax.legend(fontsize=7); ax.grid(True); ax.set_xlabel('t (s)')

    # Row 2
    t7, f = get(data, "/uav_0/debug/output/f")
    ax = axes1[2, 0]
    mg = 1.806 * 9.8065
    if t7 is not None:
        ax.plot(t7, f[:, 0], color='tab:brown', label='f')
        ax.axhline(mg, color='gray', linestyle='--', label=f'mg={mg:.1f} N')
        ax.axhline(0,  color='k',    linestyle='--', alpha=0.3)
    ax.set_title('Total thrust f (N)'); ax.legend(fontsize=7); ax.grid(True); ax.set_xlabel('t (s)')

    t8, thr = get(data, "/uav_0/debug/output/normalized_throttle")
    ax = axes1[2, 1]
    rotor_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    if t8 is not None:
        for i in range(thr.shape[1]):
            ax.plot(t8, thr[:, i], color=rotor_colors[i], label=f'rotor{i}')
        ax.axhline(0.0, color='k', linestyle='--', alpha=0.4)
        ax.axhline(1.0, color='k', linestyle=':',  alpha=0.4)
    ax.set_title('Normalized throttle'); ax.legend(fontsize=7); ax.grid(True); ax.set_xlabel('t (s)')

    t9, w = get(data, "/uav_0/debug/state/omega")
    ax = axes1[2, 2]
    if t9 is not None:
        for i, lb in enumerate(['ωx', 'ωy', 'ωz']):
            ax.plot(t9, w[:, i], color=colors[i], label=lb)
    ax.set_title('Body angular velocity ω (rad/s)'); ax.legend(fontsize=7); ax.grid(True); ax.set_xlabel('t (s)')

    fig1.tight_layout()

    # ── Figure 2: Throttle health ──────────────────────────────────────────────
    if t8 is not None:
        fig2, axes2 = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        fig2.suptitle('Per-Rotor Throttle Detail', fontsize=12, fontweight='bold')
        for i in range(4):
            axes2[i].plot(t8, thr[:, i], color=rotor_colors[i])
            axes2[i].axhline(0.0, color='k', linestyle='--', alpha=0.4, label='0')
            axes2[i].axhline(1.0, color='k', linestyle=':',  alpha=0.4, label='1')
            axes2[i].set_ylabel(f'rotor {i}')
            axes2[i].grid(True)
            neg_frac = (thr[:, i] < 0).mean() * 100
            sat_frac  = (thr[:, i] > 1).mean() * 100
            axes2[i].set_title(f'rotor {i}  —  {neg_frac:.1f}% below 0,  {sat_frac:.1f}% above 1', fontsize=9)
        axes2[-1].set_xlabel('t (s)')
        fig2.tight_layout()

    out_dir = Path(".")
    fig1.savefig(out_dir / f"{bag_name}_overview.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_dir / f'{bag_name}_overview.png'}")
    if t8 is not None:
        fig2.savefig(out_dir / f"{bag_name}_throttle.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {out_dir / f'{bag_name}_throttle.png'}")
    plt.show()


if __name__ == '__main__':
    main()