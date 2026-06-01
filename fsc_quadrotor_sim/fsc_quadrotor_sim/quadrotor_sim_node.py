import rclpy
import numpy as np
from collections import deque

from rclpy.node import Node

# Messages
from geometry_msgs.msg import PoseStamped, TwistStamped, AccelStamped, Vector3Stamped
from std_msgs.msg import Float64

# QOS
from rclpy.qos import qos_profile_sensor_data

# From this package
from .quadrotor_dynamics import QuadrotorDynamics


class QuadrotorSimNode(Node, QuadrotorDynamics):

    NUM_ROTORS = 4

    def __init__(self):
        Node.__init__(self, 'quadrotor_sim_node')
        self._declare_parameters()
        self._init_dynamics()
        self._init_state()
        self._create_subscriptions()
        self._create_publishers()

    # ------------------------------------------------------------------ init

    def _declare_parameters(self):
        self.declare_parameter('publish_rate', 250.0)
        self.declare_parameter('g', 9.8065)
        self.declare_parameter('m', 1.806)
        self.declare_parameter('J_xx', 0.016)
        self.declare_parameter('J_yy', 0.017)
        self.declare_parameter('J_zz', 0.024)
        self.declare_parameter('d', 0.215)
        self.declare_parameter('k_f', 9.7237)
        self.declare_parameter('k_m', 8e-2)
        self.declare_parameter('x0',     [0.0, 0.0, 0.0])
        self.declare_parameter('v0',     [0.0, 0.0, 0.0])
        self.declare_parameter('q0',     [0.0, 0.0, 0.0, 1.0])
        self.declare_parameter('omega0', [0.0, 0.0, 0.0])
        self.declare_parameter('input_delay', 0.026)
        self.declare_parameter('integrator', 'rk4')
        self.get_logger().info("Parameters declared. Waiting for parameter server...")

    def _init_dynamics(self):
        self._rate  = self.get_parameter('publish_rate').value
        self._g     = self.get_parameter('g').value
        self._m_    = self.get_parameter('m').value
        J_xx        = self.get_parameter('J_xx').value
        J_yy        = self.get_parameter('J_yy').value
        J_zz        = self.get_parameter('J_zz').value
        d           = self.get_parameter('d').value
        k_f         = self.get_parameter('k_f').value
        k_m         = self.get_parameter('k_m').value

        J = np.diag([J_xx, J_yy, J_zz])

        alloc_matrix = np.array([
            [ k_f,                      k_f,                      k_f,                      k_f],
            [-k_f*d/np.sqrt(2),  k_f*d/np.sqrt(2),  k_f*d/np.sqrt(2), -k_f*d/np.sqrt(2)],
            [-k_f*d/np.sqrt(2),  k_f*d/np.sqrt(2), -k_f*d/np.sqrt(2),  k_f*d/np.sqrt(2)],
            [-k_m,                     -k_m,                      k_m,                      k_m],
        ])

        QuadrotorDynamics.__init__(
            self,
            m=self._m_,
            g=self._g,
            J=J,
            alloc_matrix=alloc_matrix,
        )

        self.get_logger().info(
            f"Dynamics initialised: rate={self._rate} Hz, m={self._m_} kg, g={self._g} m/s², "
            f"J=[{J_xx}, {J_yy}, {J_zz}] kg·m², d={d} m, k_f={k_f} N, k_m={k_m} N·m"
        )

    def _init_state(self):
        x0     = np.array(self.get_parameter('x0').value,     dtype=float)
        v0     = np.array(self.get_parameter('v0').value,     dtype=float)
        q0     = np.array(self.get_parameter('q0').value,     dtype=float)
        omega0 = np.array(self.get_parameter('omega0').value, dtype=float)
        q0 /= np.linalg.norm(q0)

        self._state = np.concatenate([x0, v0, q0, omega0])
        self._throttle = np.zeros(self.NUM_ROTORS)
        self._applied_throttle = np.zeros(self.NUM_ROTORS)

        self._input_delay = self.get_parameter('input_delay').value
        self._throttle_buffer: deque = deque()  # (apply_at_sec, throttle)

        integrator = self.get_parameter('integrator').value.lower()
        if integrator == 'euler':
            self._integrate = self.euler_step
        elif integrator == 'rk4':
            self._integrate = self.rk4_step
        else:
            raise ValueError(f"Unknown integrator '{integrator}'. Choose 'rk4' or 'euler'.")

        # For jerk: finite-difference of acceleration
        self._prev_accel = self.acceleration(self._state, self._throttle)
        self._dt = 1.0 / self._rate

        self.get_logger().info(
            f"Input delay: {self._input_delay * 1e3:.1f} ms, integrator: {integrator}"
        )

    def _create_subscriptions(self):
        self._rotor_subs = [
            self.create_subscription(
                Float64,
                f'control/rotor{i}/ref',
                self._make_rotor_callback(i),
                10,
            )
            for i in range(self.NUM_ROTORS)
        ]

    def _make_rotor_callback(self, idx: int):
        def callback(msg: Float64):
            self._throttle[idx] = float(msg.data)
        return callback

    def _create_publishers(self):
        self._pose_pub  = self.create_publisher(PoseStamped,    'state/pose',           qos_profile_sensor_data)
        self._twist_pub = self.create_publisher(TwistStamped,   'state/twist',          qos_profile_sensor_data)
        self._twist_inertial_pub = self.create_publisher(TwistStamped, 'state/twist_inertial', qos_profile_sensor_data)
        self._accel_pub = self.create_publisher(AccelStamped,   'state/accel',          qos_profile_sensor_data)
        self._jerk_pub  = self.create_publisher(Vector3Stamped, 'state/jerk',           qos_profile_sensor_data)

        self._timer = self.create_timer(self._dt, self._timer_callback)

    # ------------------------------------------------------------ timer loop

    def _timer_callback(self):
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        self._throttle_buffer.append((now_sec + self._input_delay, self._throttle.copy()))

        while self._throttle_buffer and self._throttle_buffer[0][0] <= now_sec:
            _, cmd = self._throttle_buffer.popleft()
            self._applied_throttle = cmd

        self._state = self._integrate(self._state, self._applied_throttle, self._dt)

        stamp = self.get_clock().now().to_msg()
        x     = self._state[0:3]
        v     = self._state[3:6]
        q     = self._state[6:10]
        omega = self._state[10:13]

        accel = self.acceleration(self._state, self._applied_throttle)
        jerk  = (accel - self._prev_accel) / self._dt
        self._prev_accel = accel

        self._publish_pose(stamp, x, q)
        self._publish_twist(stamp, omega)
        self._publish_twist_inertial(stamp, v)
        self._publish_accel(stamp, accel)
        self._publish_jerk(stamp, jerk)

    # --------------------------------------------------------- publish helpers

    def _publish_pose(self, stamp, x, q):
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = 'world'
        msg.pose.position.x  = float(x[0])
        msg.pose.position.y  = float(x[1])
        msg.pose.position.z  = float(x[2])
        msg.pose.orientation.x = float(q[0])
        msg.pose.orientation.y = float(q[1])
        msg.pose.orientation.z = float(q[2])
        msg.pose.orientation.w = float(q[3])
        self._pose_pub.publish(msg)

    def _publish_twist(self, stamp, omega):
        # body-frame angular velocity (matches state/twist convention)
        msg = TwistStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = 'base_link'
        msg.twist.angular.x = float(omega[0])
        msg.twist.angular.y = float(omega[1])
        msg.twist.angular.z = float(omega[2])
        self._twist_pub.publish(msg)

    def _publish_twist_inertial(self, stamp, v):
        msg = TwistStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = 'world'
        msg.twist.linear.x = float(v[0])
        msg.twist.linear.y = float(v[1])
        msg.twist.linear.z = float(v[2])
        self._twist_inertial_pub.publish(msg)

    def _publish_accel(self, stamp, accel):
        msg = AccelStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = 'world'
        msg.accel.linear.x = float(accel[0])
        msg.accel.linear.y = float(accel[1])
        msg.accel.linear.z = float(accel[2])
        self._accel_pub.publish(msg)

    def _publish_jerk(self, stamp, jerk):
        msg = Vector3Stamped()
        msg.header.stamp = stamp
        msg.header.frame_id = 'world'
        msg.vector.x = float(jerk[0])
        msg.vector.y = float(jerk[1])
        msg.vector.z = float(jerk[2])
        self._jerk_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = QuadrotorSimNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
