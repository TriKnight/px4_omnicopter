from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    config_1 = os.path.join(
      get_package_share_directory('px4_omnicopter'),
      'config', 'uav_parameters',
      'x500_param.yaml'
      )
    return LaunchDescription([
        Node(
            package='px4_omnicopter',
            executable='lowlevel_attitude_setpoint',
            name='lowlevel_attitude_setpoint',
            output='screen',
            parameters=[config_1]
        ),
    ])
