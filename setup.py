from setuptools import find_packages, setup

package_name = 'px4_omnicopter'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/px4_takeoff.launch.py']),  # add launch file,
        ('share/' + package_name + '/launch', ['launch/lowlevel_attitude_setpoint.launch.py']),  # add launch file
        ('share/' + package_name + '/launch', ['launch/circle_publisher.launch.py']),  # add launch file
        ('share/' + package_name + '/launch', ['launch/example_attitude_control.launch.py']),  # add launch file
        ('share/' + package_name + '/launch', ['launch/thrush_torque_control.launch.py']),  # add launch file


    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='TriKnight',
    maintainer_email='tribien.robotics@gmail.com',
    description='PX4 Offboard takeoff example',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'px4_takeoff = src.px4_takeoff:main',
        'lowlevel_attitude_setpoint = src.lowlevel_attitude_setpoint:main',
        'circle_publisher = src.circle_publisher:main',
        'example_attitude_control = src.example_attitude_control:main',
        'thrush_torque_control = src.thrush_torque_control:main'
    ],
},
)
