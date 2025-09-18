import rclpy
from px4_mission import Px4Mission
import numpy as np
import time
from math import sin, cos, sqrt, atan2

def main():
    rclpy.init()
    mission_node = Px4Mission()
    mission_node.takeoff_prep()
    #  Take off
    for _ in range(400):
        mission_node.publish_trajectory(0.0, 0.0, -2.0, 0.0)
        rclpy.spin_once(mission_node, timeout_sec=0.05)  # 20 Hz

    # # Fly square trajectory
    # for _ in range(200):
    #     mission_node.fly_square()
    #     rclpy.spin_once(mission_node, timeout_sec=0.05)  # 20 Hz

    # for _ in range(200):
    #     mission_node.publish_attitude_setpoint(thrust_input=-0.9)
    #     rclpy.spin_once(mission_node, timeout_sec=0.05)  # 20 Hz

    ref_pos = np.array([0.0, 0.0, -3.0])
    ref_vel = np.zeros(3)
    ref_acc = np.zeros(3)
    ref_yaw = 0.0
    ref_yaw_rate = 0.0

 
    for _ in range(400):
        mission_node.publish_torque_thrush_control(ref_pos, ref_vel, ref_acc, ref_yaw, ref_yaw_rate)
        rclpy.spin_once(mission_node, timeout_sec=0.01)  # 20 Hz

    # # Trigger landing manually
    mission_node.land()
    mission_node.disarm()
    mission_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
