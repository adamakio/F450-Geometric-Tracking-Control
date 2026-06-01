from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    uav_prefix = LaunchConfiguration('uav_prefix')

    config = os.path.join(
        get_package_share_directory('fsc_quadrotor_sim'),
        'config',
        'params_quadrotor_sim.yaml'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'uav_prefix',
            default_value='uav_0',
            description='Namespace of the UAV'
        ),
        Node(
            package='fsc_quadrotor_sim',
            executable='quadrotor_sim_node',
            name='quadrotor_sim_node',
            output='screen',
            parameters=[config],
            namespace=uav_prefix,
        )
    ])
