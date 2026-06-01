"""Launch the geometric control trajectory publisher node."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Create the launch description for the trajectory publisher node."""
    default_config = os.path.join(
        get_package_share_directory('fsc_geometric_control_trajectories'),
        'config',
        'trajectory.yaml',
    )

    config_file = LaunchConfiguration('config_file')
    uav_prefix = LaunchConfiguration('uav_prefix')

    return LaunchDescription([
        DeclareLaunchArgument(
            'uav_prefix',
            default_value='uav_0',
            description='Namespace of the UAV',
        ),
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config,
            description='Path to the trajectory parameter YAML file',
        ),
        Node(
            package='fsc_geometric_control_trajectories',
            executable='trajectory_publisher',
            name='trajectory_publisher',
            namespace=uav_prefix,
            output='screen',
            parameters=[config_file],
        ),
    ])
