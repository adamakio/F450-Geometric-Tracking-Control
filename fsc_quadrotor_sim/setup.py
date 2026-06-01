from setuptools import find_packages, setup

package_name = 'fsc_quadrotor_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/quadrotor_sim_launch.py']),
        ('share/' + package_name + '/config', ['config/params_quadrotor_sim.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fsc-jupiter',
    maintainer_email='enoch.lo@mail.utoronto.ca',
    description='Quadrotor dynamics simulator using RK4 integration',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'quadrotor_sim_node = fsc_quadrotor_sim.quadrotor_sim_node:main',
        ],
    },
)
