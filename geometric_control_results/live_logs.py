#!/usr/bin/env python3
"""
Live logging and plotting of geometric controller debug topics from ROS2.

This script subscribes to the specified topics, logs their data in real-time,
and updates live plots similar to plot_hover_debug.py.
"""

import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from geometry_msgs.msg import Vector3, Vector3Stamped, Quaternion, QuaternionStamped
from std_msgs.msg import Float64, Float64MultiArray
import time
from scipy.spatial.transform import Rotation

# TODO: Import custom message types if needed, e.g.:
# from fsc_autopilot_msgs.msg import DebugVector, DebugQuaternion, DebugFloat, DebugThrottle

def get(data, topic):
    """Return (t, arr) or (None, None) if missing/empty."""
    d = data.get(topic, {})
    t = d.get("t", [])
    arr = d.get("data", [])
    if len(t) == 0:
        return None, None
    return np.asarray(list(t)), np.asarray(list(arr))

def quat_to_euler_deg(q_arr):
    """Convert (N,4) [x,y,z,w] quaternions to (N,3) Euler angles ZYX in degrees."""
    return Rotation.from_quat(q_arr).as_euler('ZYX', degrees=True)[:, ::-1]  # roll, pitch, yaw

def parse_vector_msg(msg):
    vec = msg.vector if hasattr(msg, "vector") else msg
    return np.array([vec.x, vec.y, vec.z])

def parse_quaternion_msg(msg):
    quat = msg.quaternion if hasattr(msg, "quaternion") else msg
    return np.array([quat.x, quat.y, quat.z, quat.w])

def add_legend_if_needed(ax):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7)

# Update functions for live plotting
def update_position(ax, data):
    ax.clear()
    t, x = get(data, "/uav_0/debug/state/x")
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
    add_legend_if_needed(ax)
    ax.grid(True)

def update_position_error(ax, data):
    ax.clear()
    t, ex = get(data, "/uav_0/debug/controller/e_x")
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    if t is not None:
        for i, lb in enumerate(['ex', 'ey', 'ez']):
            ax.plot(t, ex[:, i], color=colors[i], label=lb)
        ax.plot(t, np.linalg.norm(ex, axis=1), 'k--', linewidth=1.5, label='‖ex‖')
    ax.set_title('Position error (m)')
    ax.set_xlabel('t (s)')
    add_legend_if_needed(ax)
    ax.grid(True)

def update_attitude(ax, data):
    ax.clear()
    t, q = get(data, "/uav_0/debug/state/q")
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    if t is not None:
        euler = quat_to_euler_deg(q)
        for i, lb in enumerate(['roll', 'pitch', 'yaw']):
            ax.plot(t, euler[:, i], color=colors[i], label=lb)
    ax.set_title('Attitude — Euler angles (deg)')
    ax.set_xlabel('t (s)')
    add_legend_if_needed(ax)
    ax.grid(True)

def update_attitude_error(ax, data):
    ax.clear()
    t, eR = get(data, "/uav_0/debug/controller/e_R")
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    if t is not None:
        for i, lb in enumerate(['eR_x', 'eR_y', 'eR_z']):
            ax.plot(t, eR[:, i], color=colors[i], label=lb)
        ax.plot(t, np.linalg.norm(eR, axis=1), 'k--', linewidth=1.5, label='‖eR‖')
    ax.set_title('Attitude error e_R (rad)')
    ax.set_xlabel('t (s)')
    add_legend_if_needed(ax)
    ax.grid(True)

def update_omega_error(ax, data):
    ax.clear()
    t, ew = get(data, "/uav_0/debug/controller/e_omega")
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    if t is not None:
        for i, lb in enumerate(['eΩ_x', 'eΩ_y', 'eΩ_z']):
            ax.plot(t, ew[:, i], color=colors[i], label=lb)
    ax.set_title('Angular rate error e_Ω (rad/s)')
    ax.set_xlabel('t (s)')
    add_legend_if_needed(ax)
    ax.grid(True)

def update_b3_des(ax, data):
    ax.clear()
    t, b3 = get(data, "/uav_0/debug/controller/b3_des")
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    if t is not None:
        for i, lb in enumerate(['b3x', 'b3y', 'b3z']):
            ax.plot(t, b3[:, i], color=colors[i], label=lb)
        ax.axhline(1.0, color='tab:green', linestyle='--', alpha=0.5, label='ideal b3z=1')
    ax.set_title('Desired b3 axis (should be ≈[0,0,1] at hover)')
    ax.set_xlabel('t (s)')
    add_legend_if_needed(ax)
    ax.grid(True)

def update_thrust(ax, data):
    ax.clear()
    t, f = get(data, "/uav_0/debug/output/f")
    mg = 1.806 * 9.8065
    if t is not None:
        ax.plot(t, f[:, 0], color='tab:brown', label='f')
        ax.axhline(mg, color='gray', linestyle='--', label=f'mg={mg:.1f} N')
        ax.axhline(0,  color='k',    linestyle='--', alpha=0.3)
    ax.set_title('Total thrust f (N)')
    ax.set_xlabel('t (s)')
    add_legend_if_needed(ax)
    ax.grid(True)

def update_throttle(ax, data):
    ax.clear()
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
    add_legend_if_needed(ax)
    ax.grid(True)

def update_omega(ax, data):
    ax.clear()
    t, w = get(data, "/uav_0/debug/state/omega")
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    if t is not None:
        for i, lb in enumerate(['ωx', 'ωy', 'ωz']):
            ax.plot(t, w[:, i], color=colors[i], label=lb)
    ax.set_title('Body angular velocity ω (rad/s)')
    ax.set_xlabel('t (s)')
    add_legend_if_needed(ax)
    ax.grid(True)

def create_overview_figure():
    fig, axes = plt.subplots(3, 3, figsize=(17, 11))
    fig.suptitle('Geometric Controller — Live Debug', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig, axes

class LiveLogger(Node):
    def __init__(self):
        super().__init__('live_logger')
        self.start_time = time.monotonic()
        self.max_samples = 5000

        # Subscribe to the stamped debug topics used by the controller runtime.
        self.topic_subscriptions = [
            {"topic": "/uav_0/debug/state/x", "type": Vector3Stamped, "parser": parse_vector_msg},
            {"topic": "/uav_0/debug/state/v", "type": Vector3Stamped, "parser": parse_vector_msg},
            {"topic": "/uav_0/debug/state/omega", "type": Vector3Stamped, "parser": parse_vector_msg},
            {"topic": "/uav_0/debug/state/q", "type": QuaternionStamped, "parser": parse_quaternion_msg},
            {"topic": "/uav_0/debug/controller/e_x", "type": Vector3Stamped, "parser": parse_vector_msg},
            {"topic": "/uav_0/debug/controller/e_R", "type": Vector3Stamped, "parser": parse_vector_msg},
            {"topic": "/uav_0/debug/controller/e_omega", "type": Vector3Stamped, "parser": parse_vector_msg},
            {"topic": "/uav_0/debug/controller/b3_des", "type": Vector3Stamped, "parser": parse_vector_msg},
            {"topic": "/uav_0/debug/output/f", "type": Float64, "parser": lambda m: np.array([m.data])},
            {"topic": "/uav_0/debug/output/normalized_throttle", "type": Float64MultiArray, "parser": lambda m: np.array(m.data)},
        ]
        self.data = {
            topic: {
                "t": deque(maxlen=self.max_samples),
                "data": deque(maxlen=self.max_samples),
            }
            for topic in {config["topic"] for config in self.topic_subscriptions}
        }
        self.fig, self.axes = create_overview_figure()
        self.plot_timer = self.create_timer(0.2, self.refresh_plots)

        # Create subscribers
        for config in self.topic_subscriptions:
            self.create_subscription(
                config["type"],
                config["topic"],
                lambda msg, t=config["topic"], p=config["parser"]: self.callback(msg, t, p),
                10
            )

        self.get_logger().info("Live logger node started. Subscribed to debug topics.")

    def callback(self, msg, topic, parser):
        timestamp = time.monotonic() - self.start_time
        data = parser(msg)
        self.data[topic]["t"].append(timestamp)
        self.data[topic]["data"].append(data)

    def refresh_plots(self):
        if not plt.fignum_exists(self.fig.number):
            self.get_logger().info("Plot window closed. Shutting down live logger.")
            rclpy.shutdown()
            return

        update_position(self.axes[0, 0], self.data)
        update_position_error(self.axes[0, 1], self.data)
        update_attitude(self.axes[0, 2], self.data)
        update_attitude_error(self.axes[1, 0], self.data)
        update_omega_error(self.axes[1, 1], self.data)
        update_b3_des(self.axes[1, 2], self.data)
        update_thrust(self.axes[2, 0], self.data)
        update_throttle(self.axes[2, 1], self.data)
        update_omega(self.axes[2, 2], self.data)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

def main(args=None):
    plt.ion()
    rclpy.init(args=args)
    node = LiveLogger()
    try:
        plt.show(block=False)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.close('all')

if __name__ == '__main__':
    main()
