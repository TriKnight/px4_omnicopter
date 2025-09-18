from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    return LaunchDescription([
        Node(
            package='px4_omnicopter',
            executable='example_attitude_control',
            name='example_attitude_control',
            output='screen',
            
        ),
    ])
