import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleAttitudeSetpoint, VehicleCommand

import math


class PX4CircleControl(Node):
    def __init__(self):
        super().__init__('px4_circle_control')

        # Publishers
        self.att_sp_pub = self.create_publisher(
            VehicleAttitudeSetpoint,
            '/fmu/in/vehicle_attitude_setpoint_v1',
            10
        )
        self.cmd_pub = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            10
        )

        # Timers
        self.timer = self.create_timer(0.02, self.publish_attitude_setpoint)  # 50Hz
        self.cmd_timer = self.create_timer(1.0, self.keep_alive_command)      # 1Hz

        self.offboard_started = False
        self.counter = 0
        self.start_time = self.get_clock().now().nanoseconds * 1e-9

    def send_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.param1 = param1
        msg.param2 = param2
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.cmd_pub.publish(msg)

    def euler_to_quaternion(self, roll, pitch, yaw):
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return [w, x, y, z]

    def publish_attitude_setpoint(self):
        msg = VehicleAttitudeSetpoint()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000  # µs

        # Time since start
        t = (self.get_clock().now().nanoseconds * 1e-9) - self.start_time

        # Circle control parameters
        radius_tilt = 0.4        # rad, tilt amplitude
        frequency = 0.2           # Hz, one circle every 5s

        roll = radius_tilt * math.cos(2 * math.pi * frequency * t)
        pitch = radius_tilt * math.sin(2 * math.pi * frequency * t)
        yaw = 0.0

        q = self.euler_to_quaternion(roll, pitch, yaw)
        msg.q_d = q
        msg.yaw_sp_move_rate = 0.0

        # Thrust to maintain altitude (tune for your drone!)
        msg.thrust_body = [0.0, 0.0, -0.8]

        self.att_sp_pub.publish(msg)

        # Switch to OFFBOARD + ARM after ~1s of setpoints
        if not self.offboard_started:
            if self.counter > 50:
                self.send_vehicle_command(command=176, param1=1.0)  # Offboard
                self.get_logger().info("Sent OFFBOARD mode request")

                self.send_vehicle_command(command=400, param1=1.0)  # Arm
                self.get_logger().info("Sent ARM request")

                self.offboard_started = True
            self.counter += 1

    def keep_alive_command(self):
        if self.offboard_started:
            self.send_vehicle_command(command=176, param1=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = PX4CircleControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
