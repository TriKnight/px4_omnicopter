#!/usr/bin/env python3
"""
px4_lowlevel_controller.py
"""

import rclpy
from rclpy.node import Node
import numpy as np
from math import sin, cos, sqrt, atan2
from px4_msgs.msg import (
    VehicleCommand,
    VehicleAttitudeSetpoint,
    OffboardControlMode,
    VehicleThrustSetpoint,
    VehicleTorqueSetpoint,
    ActuatorMotors,
    VehicleOdometry,
    VehicleStatus
)
from geometry_msgs.msg import PoseStamped

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# QoS for PX4 topics: BEST_EFFORT
qos_profile = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)

# --- helpers -----------------------------------------------------------------
def quat_to_rotmat(q):
    """Convert quaternion [w,x,y,z] to rotation matrix (3x3)."""
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y+z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),   1 - 2*(x*x+z*z),       2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w), 1 - 2*(x*x+y*y)]
    ])
    return R

def quaternion_from_rotmat(R):
    """Convert rotation matrix to quaternion [w,x,y,z]."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    else:
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            s = 2.0 * sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            w = (R[2,1] - R[1,2]) / s
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = 2.0 * sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            w = (R[0,2] - R[2,0]) / s
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
        else:
            s = 2.0 * sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            w = (R[1,0] - R[0,1]) / s
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z])

def rotvec_FLD_to_FRD(v):
    """Rotate vector from FLU to FRD frame (flip y/z)."""
    return np.array([v[0], -v[1], -v[2]])

# --- Controller Node --------------------------------------------------------
class Px4LowlevelController(Node):
    def __init__(self):
        super().__init__('px4_lowlevel_controller')

            # --- Retrieve UAV parameters dict directly (YAML supplies it) ---
        uav_params = self.get_parameter_or('uav_parameters', None)
        if uav_params is None or not isinstance(uav_params.value, dict):
            self.get_logger().warn("No UAV parameters loaded from YAML, using defaults")
            uav_params = {}
        else:
            uav_params = uav_params.value

        # --- Assign values with fallbacks ---
        self.mass = float(uav_params.get('mass', 2.0))
        self.arm_length = float(uav_params.get('arm_length', 0.25))
        self.num_of_arms = int(uav_params.get('num_of_arms', 4))
        self.moment_constant = float(uav_params.get('moment_constant', 0.016))
        self.thrust_constant = float(uav_params.get('thrust_constant', 8.54858e-06))
        self.max_rotor_speed = float(uav_params.get('max_rotor_speed', 1000))
        self.gravity = float(uav_params.get('gravity', 9.81))
        self.PWM_MIN = int(uav_params.get('PWM_MIN', 1075))
        self.PWM_MAX = int(uav_params.get('PWM_MAX', 1950))
        self.input_scaling = float(uav_params.get('input_scaling', 1000))
        self.zero_position_armed = float(uav_params.get('zero_position_armed', 10))

        # --- Nested inertia dict ---
        inertia_params = uav_params.get('inertia', {})
        ix = float(inertia_params.get('x', 0.08612))
        iy = float(inertia_params.get('y', 0.08962))
        iz = float(inertia_params.get('z', 0.16088))
        self.inertia = np.diag([ix, iy, iz])

        # --- Nested omega_to_pwm_coefficient dict ---
        omega_params = uav_params.get('omega_to_pwm_coefficient', {})
        self.omega_to_pwm_coeff = {
            'x_2': float(omega_params.get('x_2', 0.001142)),
            'x_1': float(omega_params.get('x_1', 0.2273)),
            'x_0': float(omega_params.get('x_0', 914.2))
        }

        self.get_logger().info(
            f"Loaded UAV params: mass={self.mass}, arm_length={self.arm_length}, inertia={self.inertia.tolist()}"
        )

        # Topics
        self.declare_parameter('topics.odometry', '/fmu/out/vehicle_odometry')
        self.declare_parameter('topics.status', '/fmu/out/vehicle_status')
        self.declare_parameter('topics.command_pose', '/command/pose')

        # Controller gains
        self.declare_parameter('control_gains.K_p', [7.0, 7.0, 6.0])
        self.declare_parameter('control_gains.K_v', [6.0, 6.0, 3.0])
        self.declare_parameter('control_gains.K_R', [3.5, 3.5, 0.3])
        self.declare_parameter('control_gains.K_w', [0.5, 0.5, 0.1])
        kp = np.array(self.get_parameter('control_gains.K_p').value, dtype=float)
        kv = np.array(self.get_parameter('control_gains.K_v').value, dtype=float)
        kR = np.array(self.get_parameter('control_gains.K_R').value, dtype=float)
        kw = np.array(self.get_parameter('control_gains.K_w').value, dtype=float)
        self.Kp = kp
        self.Kv = kv
        self.KR = kR
        self.Kw = kw

        # state from topics
        self.current_pos = np.zeros(3)
        self.current_vel = np.zeros(3)
        self.current_q = np.array([1.0, 0.0, 0.0, 0.0])  # w,x,y,z
        self.current_omega = np.zeros(3)
        self.current_status = None

        # reference (command)
                # in __init__ after defining self.ref_pos etc.
        self.hover_altitude = 2.0   # [m] hover at 2 meters
        self.ref_pos = np.array([0.0, 0.0, -self.hover_altitude])  # NED frame (z negative up)
        self.ref_vel = np.zeros(3)
        self.ref_acc = np.zeros(3)
        self.ref_yaw = 0.0
        self.ref_yaw_rate = 0.0
        self.ref_quat = np.array([1.0,0.0,0.0,0.0])

        # Choose control mode
        self.control_mode = 2

        # publishers & subscribers (use BEST_EFFORT QoS for PX4)
        odom_topic = self.get_parameter('topics.odometry').value
        status_topic = self.get_parameter('topics.status').value
        cmd_pose_topic = self.get_parameter('topics.command_pose').value

        self.odom_sub = self.create_subscription(VehicleOdometry, odom_topic, self.odometry_callback, qos_profile)
        self.status_sub = self.create_subscription(VehicleStatus, status_topic, self.status_callback, qos_profile)
        self.pose_sub = self.create_subscription(PoseStamped, cmd_pose_topic, self.command_pose_callback, qos_profile)

        self.att_set_pub = self.create_publisher(VehicleAttitudeSetpoint, '/fmu/in/vehicle_attitude_setpoint_v1', qos_profile)
        self.actuator_pub = self.create_publisher(ActuatorMotors, '/fmu/in/actuator_motors', qos_profile)
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.vcmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        self.thrust_pub = self.create_publisher(VehicleThrustSetpoint, '/fmu/in/vehicle_thrust_setpoint', qos_profile)
        self.torque_pub = self.create_publisher(VehicleTorqueSetpoint, '/fmu/in/vehicle_torque_setpoint', qos_profile)

        # precompute control allocation matrices
        self._build_control_allocation()

        # Startup sequence counters & flags
        self.startup_counter = 0
        self.requested_arm = False
        self.requested_offboard = False
        # thresholds (in loop iterations at startup_publish_rate)
        self.startup_publish_rate = 20.0  # Hz
        self.arm_after = 40               # ~2s if 20Hz (increase safety)
        self.offboard_after = 80          # ~4s if 20Hz

        # timers
        self.startup_timer = self.create_timer(1.0/self.startup_publish_rate, self.publish_startup_setpoints)
        self.offboard_timer = self.create_timer(0.33, self.publish_offboard_mode)
        self.ctrl_timer = self.create_timer(0.002, self.update_controller_output)  # 100 Hz controller loop

        self.get_logger().info("PX4 low-level controller node started (with safe startup sequence)")

    # ------------------ topic callbacks -------------------------------------
    def odometry_callback(self, msg: VehicleOdometry):
        # Use fields as per px4_msgs VehicleOdometry
        self.current_pos = np.array(msg.position)          # [x, y, z]
        self.current_vel = np.array(msg.velocity)          # [vx, vy, vz]
        self.current_q = np.array(msg.q)                   # [w, x, y, z]
        self.current_omega = np.array(msg.angular_velocity)  # [wx, wy, wz]
        self.get_logger().debug(
            f"Odometry updated: pos={self.current_pos}, vel={self.current_vel}, quat={self.current_q}, omega={self.current_omega}"
        )

    def status_callback(self, msg: VehicleStatus):
        self.current_status = msg

    def command_pose_callback(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        self.ref_pos = np.array([p.x, p.y, p.z])
        self.ref_quat = np.array([q.w, q.x, q.y, q.z])
        w,x,y,z = self.ref_quat
        self.ref_yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        self.get_logger().info(f"Got first pose command, position: {p}, quaternion: {q}")

    # ------------------ matrix setup ---------------------------------------
    def _build_control_allocation(self):
        kDegToRad = np.pi/180.0
        if self.num_of_arms == 4:
            kS = np.sin(45 * kDegToRad)
            rotor_velocities_to_torques_and_thrust = np.array([
                [-kS,  kS,  kS, -kS],
                [-kS,  kS, -kS,  kS],
                [-1.0, -1.0, 1.0, 1.0],
                [1.0,  1.0, 1.0, 1.0]
            ], dtype=float)
            kdiag = np.array([
                self.thrust_constant * self.arm_length,
                self.thrust_constant * self.arm_length,
                self.moment_constant * self.thrust_constant,
                self.thrust_constant
            ], dtype=float)
            rotor_velocities_to_torques_and_thrust = (kdiag[:,None] * rotor_velocities_to_torques_and_thrust)
            self.torques_thrust_to_rotor_velocities = np.linalg.pinv(rotor_velocities_to_torques_and_thrust)
            self.throttles_to_normalized = np.array([
                [-0.5718,    0.4376,    0.5718,   -0.4376],
                [-0.3536,    0.3536,   -0.3536,    0.3536],
                [-0.2832 ,   -0.2832 ,  0.2832 ,  0.2832],
                [0.2500 ,   0.2500 ,   0.2500 ,   0.2500]
            ], dtype=float)
        else:
            self.get_logger().warn("Only 4-arm mixing currently implemented")
            self.torques_thrust_to_rotor_velocities = np.zeros((4,4))
            self.throttles_to_normalized = np.zeros((4,4))

    # ------------------ controller core ------------------------------------
    def calculate_controller_output(self):
        R_B_W = quat_to_rotmat(self.current_q)
        e_p = self.current_pos - self.ref_pos
        e_v = self.current_vel - self.ref_vel

        I_a_d = -(self.Kp * e_p) - (self.Kv * e_v) \
                + self.mass * self.gravity * np.array([0.0, 0.0, 1.0]) \
                + self.mass * self.ref_acc
        thrust = float(np.dot(I_a_d, R_B_W[:, 2]))

        # --- Desired orientation ---
        B_z_d = I_a_d.copy()
        norm = np.linalg.norm(B_z_d)
        if norm < 1e-6:
            B_z_d = np.array([0.0, 0.0, 1.0])
        else:
            B_z_d /= norm

        B_x_d = np.array([cos(self.ref_yaw), sin(self.ref_yaw), 0.0])
        B_y_d = np.cross(B_z_d, B_x_d)
        if np.linalg.norm(B_y_d) < 1e-6:
            B_y_d = np.array([0.0, 1.0, 0.0])
        else:
            B_y_d /= np.linalg.norm(B_y_d)

        R_d_w = np.zeros((3, 3))
        R_d_w[:, 0] = np.cross(B_y_d, B_z_d)
        R_d_w[:, 1] = B_y_d
        R_d_w[:, 2] = B_z_d

        # --- Attitude error ---
        e_R_mat = 0.5 * (R_d_w.T @ R_B_W - R_B_W.T @ R_d_w)
        e_R = np.array([e_R_mat[2, 1], e_R_mat[0, 2], e_R_mat[1, 0]])

        omega_ref = self.ref_yaw_rate * np.array([0.0, 0.0, 1.0])
        e_omega = self.current_omega - (R_B_W.T @ (R_d_w @ omega_ref))

        # --- Real inertia (not identity) ---
        tau = -(self.KR * e_R) - (self.Kw * e_omega) \
            + np.cross(self.current_omega, self.inertia @ self.current_omega)

        controller_output = np.zeros(4)
        controller_output[0:3] = tau
        controller_output[3] = thrust
        return controller_output, quaternion_from_rotmat(R_d_w)


    # ------------------ control allocation (inverse) -------------------------
    def px4_inverse_sitl(self, wrench):
        omega_sq = self.torques_thrust_to_rotor_velocities.dot(wrench)
        omega_sq = np.clip(omega_sq, 0.0, None)
        omega = np.sqrt(omega_sq)
        throttles = (omega - (self.zero_position_armed * np.ones_like(omega))) / max(1e-6, self.input_scaling)
        normalized = self.throttles_to_normalized.dot(throttles)
        return normalized, throttles

    # ------------------ publishers for outputs --------------------------------
    def publish_attitude_setpoint(self, collective, desired_quat):
        """
        collective: float in [0..1] meaning fraction of max thrust (positive).
        desired_quat: array [w,x,y,z]
        """
        msg = VehicleAttitudeSetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds // 1000)
        msg.q_d = [
            float(desired_quat[0]),
            float(desired_quat[1]),
            float(desired_quat[2]),
            float(desired_quat[3])
        ]

        # convert collective (positive) to PX4 NED negative Z convention
        # clamp small -> avoid zero which PX4 might treat as invalid
        coll = float(np.clip(collective, 0.0, 1.0))
        if coll < 0.02:
            coll = 0.02

        msg.thrust_body = [0.0, 0.0, -coll]
        msg.yaw_sp_move_rate = 0.0
        self.att_set_pub.publish(msg)

        # # also publish VehicleThrustSetpoint (same sign)
        tmsg = VehicleThrustSetpoint()
        tmsg.timestamp = msg.timestamp
        tmsg.timestamp_sample = msg.timestamp
        tmsg.xyz = [0.0, 0.0, -coll]
        self.thrust_pub.publish(tmsg)

        # log occasionally for debugging
        self.get_logger().info(f"AttitudeSetpoint: collective={coll:.3f}, q_d={msg.q_d}")



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

    def publish_actuator_motors(self, throttles):
        arr = [float(throttles[i]) if i < len(throttles) else float('nan') for i in range(12)]
        msg = ActuatorMotors()
        msg.control = arr
        msg.reversible_flags = 0
        ts = int(self.get_clock().now().nanoseconds // 1000)
        msg.timestamp = ts
        msg.timestamp_sample = ts
        self.actuator_pub.publish(msg)

    # ------------------ startup: stream setpoints then arm & set offboard ----
        # ------------------ startup: stream setpoints then arm & set offboard ----
    def publish_startup_setpoints(self):
        now = int(self.get_clock().now().nanoseconds // 1000)

        # publish OffboardControlMode
        mode_msg = OffboardControlMode()
        mode_msg.timestamp = now
        mode_msg.position = False
        mode_msg.velocity = False
        mode_msg.acceleration = False
        mode_msg.attitude = True
        mode_msg.body_rate = False
        self.offboard_pub.publish(mode_msg)

        # --- startup sequence: arm then request offboard ---
        self.startup_counter += 1

        if (not self.requested_arm) and (self.startup_counter >= self.arm_after):
            self.get_logger().info("Startup: sending ARM command")
            self.send_vehicle_command(400, param1=1.0)  # MAV_CMD_COMPONENT_ARM_DISARM
            self.requested_arm = True

        if (not self.requested_offboard) and (self.startup_counter >= self.offboard_after):
            self.get_logger().info("Startup: requesting OFFBOARD mode")
            self.send_vehicle_command(176, param1=1.0, param2=6.0)  # MAV_CMD_DO_SET_MODE, custom=6=Offboard
            self.requested_offboard = True

        if self.requested_offboard and self.startup_counter >= (self.offboard_after + 10):
            self.get_logger().info("Startup streaming complete; stopping startup publisher.")
            try:
                self.startup_timer.cancel()
            except Exception:
                pass



    # ------------------ periodic tasks ---------------------------------------
    def publish_offboard_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds // 1000)
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = True
        msg.body_rate = False
        self.offboard_pub.publish(msg)


    def update_controller_output(self):
        # Calculate controller outputs
        controller_output, desired_quat = self.calculate_controller_output()
        if self.current_status is None:
            return
        
        # Placeholder: implement real hardware mixing if needed
        normalized, throttles = self.px4_inverse_sitl(controller_output)
        self.get_logger().info(f"Computed wrench = {normalized:.3f}")

        # Only publish if vehicle is OFFBOARD
        self.get_logger().info(f"STARTING OFFBOARD MODE")
        if self.current_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.get_logger().info(f"OFFBOARD MODE START")
            if self.control_mode == 1:
                # Mode 1: Attitude setpoint (quat + normalized thrust)
                self.publish_attitude_setpoint(normalized[3], desired_quat)
                #self.get_logger().info(f"Publish attitude setpoint = {normalized[3]:.3f}")
            elif self.control_mode == 2:
                # Mode 2: Thrust + torque setpoints
                self.publish_thrust_torque(normalized)
            elif self.control_mode == 3:
                # Mode 3: Direct motor commands
                self.publish_actuator_motors(throttles)
            else:
                # Default: fall back to attitude mode
                self.publish_attitude_setpoint(normalized[3], desired_quat)


    # ------------------ utility to send vehicle command -----------------------
    def send_vehicle_command(self, command, param1=0.0, param2=0.0, param7=0.0):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds // 1000)
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.param7 = float(param7)
        msg.command = int(command)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.vcmd_pub.publish(msg)

# ------------------ main -----------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = Px4LowlevelController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
