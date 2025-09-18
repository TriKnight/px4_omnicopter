from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='px4_omnicopter',
            executable='px4_takeoff',
            name='px4_takeoff',
            output='screen',
        ),
    ])
