import rclpy
from rclpy.node import Node
import time

from px4_msgs.msg import (
    VehicleThrustSetpoint,
    VehicleTorqueSetpoint,
    VehicleCommand,
    OffboardControlMode
)


class PX4RawThrustTorque(Node):
    def __init__(self):
        super().__init__('px4_raw_thrust_torque')

        # Publishers
        self.thrust_pub = self.create_publisher(VehicleThrustSetpoint, '/fmu/in/vehicle_thrust_setpoint', 10)
        self.torque_pub = self.create_publisher(VehicleTorqueSetpoint, '/fmu/in/vehicle_torque_setpoint', 10)
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)

        # Timers
        self.timer = self.create_timer(0.02, self.publish_raw_control)  # 50 Hz
        self.cmd_timer = self.create_timer(1.0, self.keep_alive_offboard)  # 1 Hz

        # State
        self.counter = 0
        self.offboard_started = False
        self.start_time = self.get_clock().now().nanoseconds * 1e-9

    def send_vehicle_command(self, command, param1=0.0):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds // 1000)
        msg.param1 = param1
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.cmd_pub.publish(msg)

    def publish_raw_control(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        t = now - self.start_time

        # -------------------------------
        # OffboardControlMode (thrust + torque)
        # -------------------------------
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(self.get_clock().now().nanoseconds // 1000)
        offboard_msg.position = False
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        offboard_msg.direct_actuator = False
        offboard_msg.thrust_and_torque = True
        self.offboard_pub.publish(offboard_msg)

        # -------------------------------
        # Thrust
        # -------------------------------
        thrust_msg = VehicleThrustSetpoint()
        thrust_msg.timestamp = offboard_msg.timestamp

        if t < 3.0:
            # Strong thrust to take off
            thrust_msg.xyz = [0.0, 0.0, -0.9]  # normalized [-1,1]
        else:
            # Hover
            thrust_msg.xyz = [0.0, 0.0, -0.35]

        self.thrust_pub.publish(thrust_msg)

        # -------------------------------
        # Torque
        # -------------------------------
        torque_msg = VehicleTorqueSetpoint()
        torque_msg.timestamp = offboard_msg.timestamp

        if t > 3.0:
            torque_msg.xyz = [0.0, -0.1, 0.0]  # small pitch forward
        else:
            torque_msg.xyz = [0.0, 0.0, 0.0]

        self.torque_pub.publish(torque_msg)

        # -------------------------------
        # Enter OFFBOARD + Arm after ~1s
        # -------------------------------
        if not self.offboard_started:
            if self.counter > 50:  # ~1s
                self.send_vehicle_command(176, param1=1.0)  # Set OFFBOARD
                self.get_logger().info("✅ Sent OFFBOARD mode request")

                time.sleep(0.1)
                self.send_vehicle_command(400, param1=1.0)  # Arm
                self.get_logger().info("✅ Sent ARM request")

                self.offboard_started = True
            self.counter += 1

    def keep_alive_offboard(self):
        if self.offboard_started:
            # Keep PX4 in OFFBOARD mode
            self.send_vehicle_command(176, param1=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = PX4RawThrustTorque()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
