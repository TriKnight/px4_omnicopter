#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from px4_msgs.msg import (
    VehicleCommand, 
    TrajectorySetpoint, 
    OffboardControlMode, 
    VehicleAttitudeSetpoint,
    VehicleThrustSetpoint,
    VehicleTorqueSetpoint,
    ActuatorMotors,
    VehicleOdometry,
    VehicleStatus
)
import time
import math
import numpy as np
# QoS for PX4 topics: BEST_EFFORT
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
qos_profile = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)
from controller.lee_geometric_controller import LeeGeometricController
from controller.utilities import euler_to_quaternion, rotvec_FLD_to_FRD

class Px4Mission(Node):
    def __init__(self, node_name='px4_mission'):
        super().__init__(node_name)

        # ------------------------------
        # Publishers
        # ------------------------------
        self.vehicle_cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)
        self.offboard_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.trajectory_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.att_sp_pub = self.create_publisher(VehicleAttitudeSetpoint, '/fmu/in/vehicle_attitude_setpoint_v1', 10)
        self.torque_pub = self.create_publisher(VehicleTorqueSetpoint, '/fmu/in/vehicle_torque_setpoint', 10)
        self.thrust_pub = self.create_publisher(VehicleThrustSetpoint, '/fmu/in/vehicle_thrust_setpoint', 10)
        # Controller
         # Subs
        self.odom_sub = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odometry_callback,
            qos_profile
        )
        self.status_sub = self.create_subscription(VehicleStatus,  '/fmu/out/vehicle_status', self.status_callback, 10)
        # timer: run control loop at 100 Hz

        # ------------------------------
        # State
        # ------------------------------
        self.armed = False
        self.offboard_set = False
        self.landing = False
        self.step = 0

        # ------------------------------
        # Circle control variables
        # ------------------------------
        self.offboard_started = False
        self.counter = 0
        self.start_time = self.get_clock().now().nanoseconds * 1e-9

         # Controller
        self.controller = LeeGeometricController(mass=1.5, gravity=9.81)
        self.kp = np.array([2.0, 2.0, 1.2])   # stronger Z axis to hold altitude
        self.kv = np.array([2.0, 2.0, 1.5])    # set close to 2*sqrt(kp)
        self.kR = np.array([2.0, 2.0, 1.0])    # stronger attitude control
        self.kw = np.array([0.3, 0.3, 0.3])    # angular damping
        # Flags
        self.ready = False

        self.current_pos = np.zeros(3)
        self.current_vel = np.zeros(3)
        self.current_q = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
        self.current_omega = np.zeros(3)


    # ------------------------------
    # Vehicle commands
    # ------------------------------
    def send_vehicle_command(self, command,
                             param1=0.0, param2=0.0, param3=0.0, param4=0.0,
                             param5=0.0, param6=0.0, param7=0.0,
                             target_system=1, target_component=1,
                             source_system=1, source_component=1,
                             from_external=True):
        msg = VehicleCommand()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000  # [µs]
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.param3 = float(param3)
        msg.param4 = float(param4)
        msg.param5 = float(param5)
        msg.param6 = float(param6)
        msg.param7 = float(param7)
        msg.command = command
        msg.target_system = target_system
        msg.target_component = target_component
        msg.source_system = source_system
        msg.source_component = source_component
        msg.from_external = bool(from_external)

        self.vehicle_cmd_pub.publish(msg)
        self.get_logger().info(f"Published VehicleCommand {command} (param1={param1}, param2={param2})")

    # ------------------ Callbacks ------------------
    def odometry_callback(self, msg: VehicleOdometry):
        self.current_pos = np.array(msg.position)
        self.current_vel = np.array(msg.velocity)
        self.current_q = np.array(msg.q)   # [w, x, y, z]
        self.current_omega = np.array(msg.angular_velocity)
        self.ready = True
        print("odometry self.current_pos", self.current_pos)

    def status_callback(self, msg: VehicleStatus):
        self.current_status = msg

    # ------------------------------
    # High-level actions
    # ------------------------------
    def arm(self):
        if not self.armed:
            self.send_vehicle_command(400, param1=1.0)  # ARM
            self.armed = True
            self.get_logger().info("Drone armed.")

    def disarm(self):
        if self.armed:
            if not self.landing:
                self.get_logger().warn("Disarm requested while not landed! Sending LAND first.")
                self.land()
                time.sleep(5)
            self.send_vehicle_command(400, param1=0.0)  # DISARM
            self.armed = False
            self.offboard_set = False
            self.landing = False
            self.get_logger().info("Drone disarmed safely.")

    def set_offboard_mode(self):
        if not self.offboard_set:
            self.send_vehicle_command(176, param1=1.0, param2=6.0)  # PX4 Offboard
            self.offboard_set = True
            self.get_logger().info("Offboard mode enabled.")

    def takeoff_prep(self):
        self.arm()
        time.sleep(3)
        self.set_offboard_mode()

    def land(self):
        if not self.landing:
            self.send_vehicle_command(21)  # MAV_CMD_NAV_LAND
            self.landing = True
            self.get_logger().info("Landing command sent.")

    # ------------------------------
    # Trajectory control
    # ------------------------------
    def publish_trajectory(self, x=0.0, y=0.0, z=-2.0, yaw=0.0):
        self.keep_offboard_mode("position")
        now = self.get_clock().now().nanoseconds // 1000
        traj_msg = TrajectorySetpoint()
        traj_msg.timestamp = now
        traj_msg.position = [x, y, z]
        traj_msg.yaw = yaw
        self.trajectory_pub.publish(traj_msg)
        self.get_logger().info(f"Published TrajectorySetpoint: x={x}, y={y}, z={z}, yaw={yaw}")

    def fly_square(self, step_size=2.0, steps_per_edge=50):
        if self.landing:
            return
        x, y, z = 0.0, 0.0, -2.0
        if self.step < steps_per_edge:
            x, y = step_size, 0.0
        elif self.step < 2 * steps_per_edge:
            x, y = step_size, step_size
        elif self.step < 3 * steps_per_edge:
            x, y = 0.0, step_size
        elif self.step < 4 * steps_per_edge:
            x, y = 0.0, 0.0

        self.publish_trajectory(x, y, z)
        self.step += 1

    # ------------------------------
    # Circle attitude control
    # ------------------------------
    

    def keep_offboard_mode(self, mode):
        now = self.get_clock().now().nanoseconds // 1000
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = now
        if mode == "position":
            offboard_msg.position = True
            offboard_msg.attitude =False
            offboard_msg.thrust_and_torque=False
        if mode == "attitude":
            offboard_msg.position = False
            offboard_msg.attitude = True
            offboard_msg.thrust_and_torque=False
        if mode == "thrush_and_torque":
            offboard_msg.position = False
            offboard_msg.attitude = False
            offboard_msg.thrust_and_torque=True
        self.offboard_mode_pub.publish(offboard_msg)

    def publish_attitude_setpoint(self, roll_input=0.0, pitch_input=0.0, yaw_input=0.0, thrust_input=-0.7):
        # Ensure OFFBOARD mode is active
        self.keep_offboard_mode("attitude")

        # Create message
        msg = VehicleAttitudeSetpoint()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000  # µs

        # Compute quaternion from input angles
        q = euler_to_quaternion(roll_input, pitch_input, yaw_input)
        msg.q_d = q
        msg.yaw_sp_move_rate = yaw_input
        msg.thrust_body = [0.0, 0.0, thrust_input]

        # Publish
        self.att_sp_pub.publish(msg)
        self.get_logger().info(
            f"Published AttitudeSetpoint: q={q}, thrust_body={msg.thrust_body}, yaw_setpoint={msg.yaw_sp_move_rate}"
        )
        # Send OFFBOARD and ARM commands once after 1 second (~50 calls at 50Hz)
        if not self.offboard_started:
            if self.counter > 50:
                self.send_vehicle_command(176, param1=1.0)  # OFFBOARD
                self.get_logger().info("Sent OFFBOARD mode request")
                self.send_vehicle_command(400, param1=1.0)  # ARM
                self.get_logger().info("Sent ARM request")
                self.offboard_started = True
            self.counter += 1
    
    def publish_torque_thrush_control(self, ref_pos, ref_vel, ref_acc, ref_yaw, ref_yaw_rate):
        # only run if odometry has updated at least once
        if not self.ready:
            self.get_logger().warn("Waiting for odometry data...")
            return

        self.keep_offboard_mode("thrush_and_torque")

        # Debug: show latest odometry state
        self.get_logger().info(
            f"Current pos={self.current_pos}, vel={self.current_vel}, q={self.current_q}, omega={self.current_omega}"
        )

        normalized, throttles, desired_quat = self.controller.calculate_controller_output(
            self.kp, self.kv, self.kR, self.kw,
            self.current_pos, self.current_vel, self.current_q, self.current_omega,
            ref_pos, ref_vel, ref_acc, ref_yaw, ref_yaw_rate
        )

        print(f"Normalized: {np.round(normalized, 4)}")
        print(f"Throttles: {np.round(throttles, 4)}")
        print(f"Desired quaternion: {np.round(desired_quat, 4)}")

        # self.get_logger().info(f"Controller out: Normalized={normalized}, throttles={throttles}")
        self.publish_thrust_torque(normalized)

     # ------------------ publishers for outputs --------------------------------
    def publish_thrust_torque(self, normalized_output):
        tmsg = VehicleThrustSetpoint()
        tqmsg = VehicleTorqueSetpoint()
        tmsg.timestamp = int(self.get_clock().now().nanoseconds // 1000)
        tqmsg.timestamp = tmsg.timestamp
        tmsg.timestamp_sample = tmsg.timestamp
        tqmsg.timestamp_sample = tqmsg.timestamp
        tmsg.xyz = [0.0, 0.0, float(-max(0.1, normalized_output[3]))]
        torque_flu = normalized_output[0:3]
        torque_frd = rotvec_FLD_to_FRD(torque_flu)
        tqmsg.xyz = [float(torque_frd[0]), float(torque_frd[1]), float(torque_frd[2])]
        self.thrust_pub.publish(tmsg)
        self.torque_pub.publish(tqmsg)

