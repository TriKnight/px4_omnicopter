from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    return LaunchDescription([
        Node(
            package='px4_omnicopter',
            executable='thrush_torque_control',
            name='thrush_torque_control',
            output='screen',
            
        ),
    ])
