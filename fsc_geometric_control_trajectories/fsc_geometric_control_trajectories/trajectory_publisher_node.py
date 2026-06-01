"""ROS 2 node that publishes desired trajectory references."""

from typing import Sequence, Tuple

from geometry_msgs.msg import Point, PointStamped, Vector3, Vector3Stamped
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

from .trajectories import (
    DesiredTrajectoryState,
    circle,
    elliptic_helix,
    figure_eight,
    hover,
    yaw_spin,
)


class TrajectoryPublisherNode(Node):
    """Publish desired trajectory states for the geometric controller."""

    def __init__(self) -> None:
        """Initialize publishers, parameters, and the periodic timer."""
        super().__init__('trajectory_publisher')

        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('trajectory_type', 'hover')
        self.declare_parameter('hover_position', [0.0, 0.0, 1.0])
        self.declare_parameter('hover_yaw', 0.0)
        self.declare_parameter('yaw_spin_rate', 0.5)
        self.declare_parameter('level_height', 1.0)
        self.declare_parameter('radius', 0.5)
        self.declare_parameter('angular_velocity', 0.5)

        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.trajectory_type = str(self.get_parameter('trajectory_type').value)
        self.hover_position = self._parse_position(
            self.get_parameter('hover_position').value
        )
        self.hover_yaw = float(self.get_parameter('hover_yaw').value)
        self.yaw_spin_rate = float(self.get_parameter('yaw_spin_rate').value)
        self.level_height = float(self.get_parameter('level_height').value)
        self.radius = float(self.get_parameter('radius').value)
        self.angular_velocity = float(
            self.get_parameter('angular_velocity').value
        )

        if self.publish_rate <= 0.0:
            raise ValueError('publish_rate must be greater than zero.')
        if self.radius < 0.0:
            raise ValueError('radius must be non-negative.')

        if self.trajectory_type not in {
            'hover', 'yaw_spin', 'circle', 'figure_eight', 'elliptic_helix'
        }:
            self.get_logger().warning(
                f"Unsupported trajectory_type '{self.trajectory_type}'; "
                'defaulting to hover.'
            )
            self.trajectory_type = 'hover'

        self.position_pub = self.create_publisher(
            PointStamped, 'trajectory/desired/position', 10
        )
        self.velocity_pub = self.create_publisher(
            Vector3Stamped, 'trajectory/desired/velocity', 10
        )
        self.acceleration_pub = self.create_publisher(
            Vector3Stamped, 'trajectory/desired/acceleration', 10
        )
        self.jerk_pub = self.create_publisher(
            Vector3Stamped, 'trajectory/desired/jerk', 10
        )
        self.snap_pub = self.create_publisher(
            Vector3Stamped, 'trajectory/desired/snap', 10
        )
        self.yaw_pub = self.create_publisher(
            Float64, 'trajectory/desired/yaw', 10
        )
        self.yaw_rate_pub = self.create_publisher(
            Float64, 'trajectory/desired/yaw_rate', 10
        )
        self.yaw_acceleration_pub = self.create_publisher(
            Float64, 'trajectory/desired/yaw_acceleration', 10
        )

        self.timer = self.create_timer(
            1.0 / self.publish_rate,
            self._publish_trajectory,
        )
        self.start_time = self.get_clock().now()

    def _parse_position(self, value: Sequence[float]) -> Tuple[float, float, float]:
        """Validate and convert the hover position parameter."""
        if len(value) != 3:
            raise ValueError('hover_position must contain exactly three values.')
        return (float(value[0]), float(value[1]), float(value[2]))

    def _vector3_msg(self, value: Tuple[float, float, float]) -> Vector3:
        """Create a `geometry_msgs/Vector3` from a 3-tuple."""
        return Vector3(x=value[0], y=value[1], z=value[2])

    def _desired_state_at(self, t: float) -> DesiredTrajectoryState:
        """Return the desired state for the configured trajectory."""
        if self.trajectory_type == 'yaw_spin':
            return yaw_spin(
                t,
                self.hover_position,
                self.hover_yaw,
                self.yaw_spin_rate,
            )
        if self.trajectory_type == 'circle':
            return circle(
                t,
                self.radius,
                self.angular_velocity,
                self.level_height,
            )
        if self.trajectory_type == 'elliptic_helix':
            return elliptic_helix(t)
        if self.trajectory_type == 'figure_eight':
            return figure_eight(
                t,
                self.radius,
                self.angular_velocity,
                self.level_height,
            )
        return hover(self.hover_position, self.hover_yaw)

    def _publish_trajectory(self) -> None:
        """Publish the current desired trajectory."""
        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds * 1e-9
        state = self._desired_state_at(t)
        stamp = now.to_msg()

        position_msg = PointStamped()
        position_msg.header.stamp = stamp
        position_msg.point = Point(
            x=state.position[0],
            y=state.position[1],
            z=state.position[2],
        )
        self.position_pub.publish(position_msg)

        velocity_msg = Vector3Stamped()
        velocity_msg.header.stamp = stamp
        velocity_msg.vector = self._vector3_msg(state.velocity)
        self.velocity_pub.publish(velocity_msg)

        acceleration_msg = Vector3Stamped()
        acceleration_msg.header.stamp = stamp
        acceleration_msg.vector = self._vector3_msg(state.acceleration)
        self.acceleration_pub.publish(acceleration_msg)

        jerk_msg = Vector3Stamped()
        jerk_msg.header.stamp = stamp
        jerk_msg.vector = self._vector3_msg(state.jerk)
        self.jerk_pub.publish(jerk_msg)

        snap_msg = Vector3Stamped()
        snap_msg.header.stamp = stamp
        snap_msg.vector = self._vector3_msg(state.snap)
        self.snap_pub.publish(snap_msg)

        self.yaw_pub.publish(Float64(data=state.yaw))
        self.yaw_rate_pub.publish(Float64(data=state.yaw_rate))
        self.yaw_acceleration_pub.publish(Float64(data=state.yaw_acceleration))


def main(args=None) -> None:
    """Spin the trajectory publisher node."""
    rclpy.init(args=args)
    node = TrajectoryPublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
