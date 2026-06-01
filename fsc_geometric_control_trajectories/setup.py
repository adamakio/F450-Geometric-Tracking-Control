from setuptools import find_packages, setup

package_name = 'fsc_geometric_control_trajectories'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/trajectory.yaml']),
        ('share/' + package_name + '/launch', ['launch/trajectory.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fsc-jupiter',
    maintainer_email='zouhamaimou@gmail.com',
    description='ROS 2 trajectory publishers for geometric control references',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'trajectory_publisher = '
            'fsc_geometric_control_trajectories.trajectory_publisher_node:main',
        ],
    },
)
