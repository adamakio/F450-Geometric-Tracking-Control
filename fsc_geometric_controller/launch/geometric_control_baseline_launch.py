from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    # Launch argument
    uav_prefix = LaunchConfiguration('uav_prefix')

    config = os.path.join(
        get_package_share_directory('fsc_geometric_controller'),
        'config',
        'params_geometric_control.yaml'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'uav_prefix',
            default_value='uav_0',
            description='Namespace of the UAV'
        ),
        Node(
            package='fsc_geometric_controller',
            executable='geometric_control_node',
            name='geometric_control_node',
            output='screen',
            parameters=[config],
            namespace=uav_prefix,
        )
    ])
