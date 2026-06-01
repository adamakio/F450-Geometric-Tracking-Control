import rclpy
import numpy as np

from typing import List, Tuple
import rclpy.logging

from rclpy.node import Node
from rclpy.publisher import Publisher

# Messages
from geometry_msgs.msg import PoseStamped, TwistStamped, AccelStamped
from geometry_msgs.msg import PointStamped, Vector3Stamped, QuaternionStamped
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray

# QOS
from rclpy.qos import qos_profile_sensor_data

# From this package
from .geometric_control import GeometricController


class GeometricControlNode(Node, GeometricController):

    def __init__(self):
        Node.__init__(self, 'geometric_control_node')
        self.debug_enabled = True
        self._declare_parameters()
        self._init_controller()
        self._init_state()
        self._create_subscriptions()
        self._create_publishers()
        self.initialize_log_publishers()

    # ------------------------------------------------------------------ init

    def _declare_parameters(self):
        self.declare_parameter('publish_rate', 250.0)
        self.declare_parameter('k_x', 1.0)
        self.declare_parameter('k_v', 1.0)
        self.declare_parameter('k_R', 1.0)
        self.declare_parameter('k_Omega', 1.0)
        self.declare_parameter('k_R_yaw', 1.0)
        self.declare_parameter('k_Omega_yaw', 1.0)
        self.declare_parameter('g', 9.8065)
        self.declare_parameter('m', 1.806)
        self.declare_parameter('J_xx', 0.016)
        self.declare_parameter('J_yy', 0.017)
        self.declare_parameter('J_zz', 0.024)
        self.declare_parameter('d', 0.215)    # distance from center to each rotor
        self.declare_parameter('k_f', 9.7237) # thrust coefficient
        self.declare_parameter('k_m', 8e-2)   # moment coefficient
        self.get_logger().info("Declared parameters with default values. Waiting for parameters to be set...")

    def _init_controller(self):
        self.rate    = self.get_parameter('publish_rate').value
        self.k_x     = self.get_parameter('k_x').value
        self.k_v     = self.get_parameter('k_v').value
        self.k_R     = self.get_parameter('k_R').value
        self.k_Omega = self.get_parameter('k_Omega').value
        self.k_R_yaw = self.get_parameter('k_R_yaw').value
        self.k_Omega_yaw = self.get_parameter('k_Omega_yaw').value
        self.K_R     = np.diag([self.k_R, self.k_R, self.k_R_yaw])
        self.K_Omega = np.diag([self.k_Omega, self.k_Omega, self.k_Omega_yaw])
        self.g       = self.get_parameter('g').value
        self.m       = self.get_parameter('m').value
        self.J_xx    = self.get_parameter('J_xx').value
        self.J_yy    = self.get_parameter('J_yy').value
        self.J_zz    = self.get_parameter('J_zz').value
        self.d       = self.get_parameter('d').value
        self.k_f     = self.get_parameter('k_f').value
        self.k_m     = self.get_parameter('k_m').value
        self.get_logger().info(
            f"Parameters loaded: publish_rate={self.rate}, k_x={self.k_x}, k_v={self.k_v}, "
            f"K_R={self.K_R}, K_Omega={self.K_Omega}, "
            f"g={self.g}, m={self.m}, "
            f"J_xx={self.J_xx}, J_yy={self.J_yy}, J_zz={self.J_zz}, "
            f"d={self.d}, k_f={self.k_f}, k_m={self.k_m}"
        )

        self.num_rotors = 4  # only quadrotors supported for now
        J = np.diag([self.J_xx, self.J_yy, self.J_zz])

        allocation_matrix = np.array([
            [ self.k_f,                          self.k_f,                         self.k_f,                         self.k_f],
            [-self.k_f*self.d/np.sqrt(2),  self.k_f*self.d/np.sqrt(2),  self.k_f*self.d/np.sqrt(2), -self.k_f*self.d/np.sqrt(2)],
            [-self.k_f*self.d/np.sqrt(2),  self.k_f*self.d/np.sqrt(2), -self.k_f*self.d/np.sqrt(2),  self.k_f*self.d/np.sqrt(2)],
            [-self.k_m,                         -self.k_m,                         self.k_m,                         self.k_m],
        ])
        alloc_inv = np.array([
            [1/(4*self.k_f), -np.sqrt(2)/(4*self.k_f*self.d), -np.sqrt(2)/(4*self.k_f*self.d), -1/(4*self.k_m)],
            [1/(4*self.k_f),  np.sqrt(2)/(4*self.k_f*self.d),  np.sqrt(2)/(4*self.k_f*self.d), -1/(4*self.k_m)],
            [1/(4*self.k_f),  np.sqrt(2)/(4*self.k_f*self.d), -np.sqrt(2)/(4*self.k_f*self.d),  1/(4*self.k_m)],
            [1/(4*self.k_f), -np.sqrt(2)/(4*self.k_f*self.d),  np.sqrt(2)/(4*self.k_f*self.d),  1/(4*self.k_m)],
        ])
        assert np.allclose(allocation_matrix @ alloc_inv, np.eye(4)), \
            "Allocation matrix and its inverse do not multiply to identity!"

        GeometricController.__init__(
            self,
            k_x=self.k_x,
            k_v=self.k_v,
            K_R=self.K_R,
            K_Omega=self.K_Omega,
            m=self.m,
            g=self.g,
            J=J,
            alloc_inv=alloc_inv,
        )

    def _init_state(self):
        # Desired trajectory
        self.x_des        = np.zeros(3)
        self.v_des        = np.zeros(3)
        self.a_des        = np.zeros(3)
        self.j_des        = np.zeros(3)
        self.s_des        = np.zeros(3)
        self.yaw_des      = 0.0  # radians
        self.yaw_rate_des = 0.0  # radians/s
        self.yaw_accel_des = 0.0 # radians/s^2

        # Vehicle state
        self.x     = np.zeros(3)
        self.q     = np.array([0.0, 0.0, 0.0, 1.0])  # quaternion (x, y, z, w)
        self.omega = np.zeros(3)
        self.v     = np.zeros(3)
        self.a     = np.zeros(3)
        self.j     = np.zeros(3)

        self._got_first_state = False
        self._got_first_trajectory = False
        self._first_callback = True

    def _create_subscriptions(self):
        self.traj_position_sub = self.create_subscription(
            PointStamped, 'trajectory/desired/position', self.update_desired_position, 10)
        self.traj_velocity_sub = self.create_subscription(
            Vector3Stamped, 'trajectory/desired/velocity', self.update_desired_velocity, 10)
        self.traj_acceleration_sub = self.create_subscription(
            Vector3Stamped, 'trajectory/desired/acceleration', self.update_desired_acceleration, 10)
        self.traj_jerk_sub = self.create_subscription(
            Vector3Stamped, 'trajectory/desired/jerk', self.update_desired_jerk, 10)
        self.traj_snap_sub = self.create_subscription(
            Vector3Stamped, 'trajectory/desired/snap', self.update_desired_snap, 10)
        self.traj_yaw_sub = self.create_subscription(
            Float64, 'trajectory/desired/yaw', self.update_desired_yaw, 10)
        self.traj_yaw_rate_sub = self.create_subscription(
            Float64, 'trajectory/desired/yaw_rate', self.update_desired_yaw_rate, 10)
        self.traj_yaw_accel_sub = self.create_subscription(
            Float64, 'trajectory/desired/yaw_acceleration', self.update_desired_yaw_acceleration, 10)
        self.pose_sub = self.create_subscription(
            PoseStamped, 'state/pose', self.update_pose, qos_profile_sensor_data)
        self.twist_sub = self.create_subscription(
            TwistStamped, 'state/twist', self.update_angular_velocity, qos_profile_sensor_data)
        self.twist_inertial_sub = self.create_subscription(
            TwistStamped, 'state/twist_inertial', self.update_linear_velocity, qos_profile_sensor_data)
        self.acceleration_sub = self.create_subscription(
            AccelStamped, 'state/accel', self.update_acceleration, qos_profile_sensor_data)
        self.jerk_sub = self.create_subscription(
            Vector3Stamped, 'state/jerk', self.update_jerk, qos_profile_sensor_data)

    def _create_publishers(self):
        self.rotor_pubs: List[Publisher] = [
            self.create_publisher(Float64, f'control/rotor{i}/ref', 10)
            for i in range(self.num_rotors)
        ]
        self.t0 = self.get_clock().now().nanoseconds * 1e-9
        self.timer = self.create_timer(1.0 / self.rate, self.timer_callback)

    def initialize_log_publishers(self):
        # Controller internals
        self.log_e_x_pub       = self.create_publisher(Vector3Stamped,    "control/internal/e_x",       10)
        self.log_e_v_pub       = self.create_publisher(Vector3Stamped,    "control/internal/e_v",       10)
        self.log_e_R_pub       = self.create_publisher(Vector3Stamped,    "control/internal/e_R",       10)
        self.log_e_omega_pub   = self.create_publisher(Vector3Stamped,    "control/internal/e_omega",   10)
        self.log_omega_des_pub = self.create_publisher(Vector3Stamped,    "control/internal/omega_des", 10)
        self.log_alpha_des_pub = self.create_publisher(Vector3Stamped,    "control/internal/alpha_des", 10)
        self.log_roll_des_pub  = self.create_publisher(Float64,           "control/internal/roll_des",  10)
        self.log_pitch_des_pub = self.create_publisher(Float64,           "control/internal/pitch_des", 10)

        # Outputs
        self.log_f_pub         = self.create_publisher(Float64,           "control/output/f",                   10)
        self.log_M_pub         = self.create_publisher(Vector3Stamped,    "control/output/M",                   10)
        self.log_forces_pub    = self.create_publisher(Float64MultiArray, "control/output/forces",              10)
        self.log_throttle_pub  = self.create_publisher(Float64MultiArray, "control/output/normalized_throttle", 10)

    # --------------------------------------------------------- message helpers

    def _vector3_msg(self, vec: np.ndarray) -> Vector3Stamped:
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.vector.x = float(vec[0])
        msg.vector.y = float(vec[1])
        msg.vector.z = float(vec[2])
        return msg

    def _quat_msg(self, quat_xyzw: np.ndarray) -> QuaternionStamped:
        msg = QuaternionStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.quaternion.x = float(quat_xyzw[0])
        msg.quaternion.y = float(quat_xyzw[1])
        msg.quaternion.z = float(quat_xyzw[2])
        msg.quaternion.w = float(quat_xyzw[3])
        return msg

    def _float_msg(self, value: float) -> Float64:
        msg = Float64()
        msg.data = float(value)
        return msg

    def _array_msg(self, arr: np.ndarray) -> Float64MultiArray:
        msg = Float64MultiArray()
        msg.data = [float(x) for x in arr]
        return msg

    def _rpy_from_rotmat(self, R: np.ndarray) -> Tuple[float, float, float]:
        pitch = float(np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2])))
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        return roll, pitch, yaw

    # -------------------------------------------------------------- log pubs

    def publish_log_controller_terms(self):
        if not self.debug_enabled:
            return
        roll_des, pitch_des, _ = self._rpy_from_rotmat(self.R_des)
        self.log_e_x_pub.publish(self._vector3_msg(self.e_x))
        self.log_e_v_pub.publish(self._vector3_msg(self.e_v))
        self.log_e_R_pub.publish(self._vector3_msg(self.e_R))
        self.log_e_omega_pub.publish(self._vector3_msg(self.e_omega))
        self.log_f_pub.publish(self._float_msg(self.f))
        self.log_M_pub.publish(self._vector3_msg(self.M))
        self.log_throttle_pub.publish(self._array_msg(self.normalized_throttle))
        self.log_omega_des_pub.publish(self._vector3_msg(self.omega_des))
        self.log_alpha_des_pub.publish(self._vector3_msg(self.alpha_des))
        self.log_roll_des_pub.publish(self._float_msg(roll_des))
        self.log_pitch_des_pub.publish(self._float_msg(pitch_des))

    # ------------------------------------------------------- state callbacks

    def update_desired_position(self, msg: PointStamped):
        self.x_des = np.array([msg.point.x, msg.point.y, msg.point.z])
        self._got_first_trajectory = True

    def update_desired_velocity(self, msg: Vector3Stamped):
        self.v_des = np.array([msg.vector.x, msg.vector.y, msg.vector.z])

    def update_desired_acceleration(self, msg: Vector3Stamped):
        self.a_des = np.array([msg.vector.x, msg.vector.y, msg.vector.z])

    def update_desired_jerk(self, msg: Vector3Stamped):
        self.j_des = np.array([msg.vector.x, msg.vector.y, msg.vector.z])

    def update_desired_snap(self, msg: Vector3Stamped):
        self.s_des = np.array([msg.vector.x, msg.vector.y, msg.vector.z])

    def update_desired_yaw(self, msg: Float64):
        self.yaw_des = msg.data

    def update_desired_yaw_rate(self, msg: Float64):
        self.yaw_rate_des = msg.data

    def update_desired_yaw_acceleration(self, msg: Float64):
        self.yaw_accel_des = msg.data

    def update_pose(self, msg: PoseStamped):
        self.x = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.q = np.array([
            msg.pose.orientation.x, msg.pose.orientation.y,
            msg.pose.orientation.z, msg.pose.orientation.w,
        ])
        self._got_first_state = True

    def update_angular_velocity(self, msg: TwistStamped):
        self.omega = np.array([
            msg.twist.angular.x,
            msg.twist.angular.y,
            msg.twist.angular.z,
        ])

    def update_linear_velocity(self, msg: TwistStamped):
        self.v = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])

    def update_acceleration(self, msg: AccelStamped):
        self.a = np.array([msg.accel.linear.x, msg.accel.linear.y, msg.accel.linear.z])

    def update_jerk(self, msg: Vector3Stamped):
        self.j = np.array([msg.vector.x, msg.vector.y, msg.vector.z])

    # --------------------------------------------------------- control loop

    def timer_callback(self):
        if not self._got_first_state or not self._got_first_trajectory:
            return

        if self._first_callback:
            self.get_logger().info("First state and trajectory received, starting control.")
            self._first_callback = False

        output = self.compute_control()
        self.publish_log_controller_terms()

        for i in range(self.num_rotors):
            rotor_msg = Float64()
            rotor_msg.data = float(np.clip(output.normalized_throttle[i], 0.0, 1.0))
            self.rotor_pubs[i].publish(rotor_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GeometricControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
