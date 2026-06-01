from setuptools import find_packages, setup

package_name = 'fsc_geometric_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/geometric_control_baseline_launch.py']),
        ('share/' + package_name + '/config', ['config/params_geometric_control.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fsc-jupiter',
    maintainer_email='zouhamaimou@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'geometric_control_node = fsc_geometric_controller.geometric_control_node:main',
        ],
    },
)
