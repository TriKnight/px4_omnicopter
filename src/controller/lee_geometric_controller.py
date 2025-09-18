#!/usr/bin/env python3
import numpy as np
from math import sin, cos
from scipy.spatial.transform import Rotation as R
from px4_msgs.msg import VehicleTorqueSetpoint, VehicleThrustSetpoint
import time

import numpy as np
from math import sin, cos
from scipy.spatial.transform import Rotation as R

# --- Utilities ---
def hat(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def vee(M):
    return np.array([M[2,1], M[0,2], M[1,0]])

def quaternion_from_rotmat(R_mat):
    """Convert rotation matrix to quaternion (w,x,y,z)."""
    return R.from_matrix(R_mat).as_quat()[[3,0,1,2]]  # reorder to w,x,y,z

# --- Controller ---
class LeeGeometricController:
    def __init__(self, mass=1.0, gravity=9.81, inertia=None):
        self.mass = mass
        self.g = gravity
        self.J = np.diag([0.08612, 0.08962, 0.16088]) if inertia is None else inertia


        # quadrotor parameters
        self.arm_length = 0.25
        self.num_of_arms = 4
        self.thrust_constant = 8.54858e-06
        self.moment_constant = 0.016
        self.PWM_MIN = 1075
        self.PWM_MAX = 1950
        self.input_scaling = 1000
        self.zero_position_armed = 10

        self.e3 = np.array([0,0,1])

        # precompute allocation matrices
        self._build_control_allocation()

    # ------------------ control allocation -------------------------
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
            rotor_velocities_to_torques_and_thrust = np.diag(kdiag) @ rotor_velocities_to_torques_and_thrust
            self.torques_thrust_to_rotor_velocities = np.linalg.pinv(rotor_velocities_to_torques_and_thrust)

            # normalized throttle mapping (empirical / from PX4)
            self.throttles_to_normalized = np.array([
                [-0.5718,  0.4376,  0.5718, -0.4376],
                [-0.3536,  0.3536, -0.3536,  0.3536],
                [-0.2832, -0.2832,  0.2832,  0.2832],
                [ 0.2500,  0.2500,  0.2500,  0.2500]
            ], dtype=float)
        else:
            self.torques_thrust_to_rotor_velocities = np.zeros((4,4))
            self.throttles_to_normalized = np.zeros((4,4))

    def px4_inverse(self, wrench):
        # wrench: [Mx, My, Mz, thrust]
        normalized_torque_and_thrust = np.zeros(4)
        omega = np.zeros(4)
        throttles = np.zeros(4)
        # np.abs is very important hack to make the omega value never go negative
        omega_sq = (self.torques_thrust_to_rotor_velocities @ wrench)
        omega_sq = np.clip(omega_sq, 0.0, None)
        omega = np.sqrt(omega_sq)

        throttles = (omega - (self.zero_position_armed * np.ones(4))) / self.input_scaling
        throttles = np.clip(throttles, 0.1, 1.0)

        normalized_torque_and_thrust = self.throttles_to_normalized @ throttles
        return normalized_torque_and_thrust, throttles

    # ------------------ main controller -------------------------
    def hat(v):
        return np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])

    def vee(M):
        return np.array([M[2,1], M[0,2], M[1,0]])
    
    def calculate_controller_output(self, kx, kv, kR, kw,
                                    current_pos, current_vel, current_q, current_omega,
                                    ref_pos, ref_vel, ref_acc, ref_yaw, ref_yaw_rate):
        """
        Geometric controller (SE(3)) with PX4 allocation
        Inputs in NED frame, +Z down
        """

        # --- Convert quaternion to rotation matrix ---
        q_xyzw = np.array([current_q[1], current_q[2], current_q[3], current_q[0]])
        R_B_W = R.from_quat(q_xyzw).as_matrix()

        # --- Position error ---
        e_p = current_pos - ref_pos
        e_v = current_vel - ref_vel

        # --- Desired force ---
        F_des = -kx*e_p - kv*e_v + self.mass*(ref_acc + np.array([0,0,self.g]))

        # --- Compute thrust ---
        thrust = np.dot(F_des, R_B_W[:,2])
        thrust = max(thrust, 0.0)

        # --- Desired orientation ---
        B_z_d = F_des / (np.linalg.norm(F_des) + 1e-9)
        B_x_c = np.array([cos(ref_yaw), sin(ref_yaw), 0.0])
        if abs(np.dot(B_x_c, B_z_d)) > 0.99:
            B_x_c = np.array([0.0, 1.0, 0.0])
        B_y_d = np.cross(B_z_d, B_x_c); B_y_d /= np.linalg.norm(B_y_d)
        B_x_d = np.cross(B_y_d, B_z_d)
        R_d_w = np.column_stack((B_x_d, B_y_d, B_z_d))

        # --- Attitude error ---
        e_R = 0.5 * vee(R_d_w.T @ R_B_W - R_B_W.T @ R_d_w)

        omega_ref = ref_yaw_rate * np.array([0,0,1])
        e_omega = current_omega - (R_B_W.T @ (R_d_w @ omega_ref))

        # --- Control torque ---
        tau = -(kR * e_R) - (kw * e_omega) + np.cross(current_omega, self.J @ current_omega)

        # --- Wrench [Mx, My, Mz, Thrust] ---
        wrench = np.zeros(4)
        wrench[0:3] = tau
        wrench[3] = thrust

        # --- Desired quaternion ---
        desired_quat = quaternion_from_rotmat(R_d_w)

        # --- Rotor-level mapping ---
        normalized, throttles = self.px4_inverse(wrench)
        return normalized, throttles, desired_quat



if __name__ == "__main__":
    controller = LeeGeometricController(mass=1.5, gravity=9.81)
    # Gains
    kp = np.array([3.0, 3.0, 4.0])   # stronger Z axis to hold altitude
    kv = np.array([3.0, 3.0, 2.5])    # set close to 2*sqrt(kp)
    kR = np.array([3.0, 3.0, 2.0])    # stronger attitude control
    kw = np.array([0.3, 0.3, 0.3])    # angular damping

    # Define test cases, these test      following NED frame
    test_cases = [
        # (current_pos, current_vel, current_q, current_omega, ref_pos, ref_vel, ref_acc, ref_yaw, ref_yaw_rate, name)
    
            
        (np.array([0.010564281605184078,
                0.03900080546736717,
                -0.09714360535144806]),              # current_pos

        np.array([0.0060882847756147385,
                0.0017290411051362753,
                -0.006534361746162176]),             # current_vel

        np.array([0.5934420228004456,
                0.005858979653567076,
                -0.007116937078535557,
                0.8048239350318909]),                # current_q [w, x, y, z]

        np.array([-0.00018392044876236469,
                0.00023749393585603684,
                -0.00027779547963291407]),           # current_omega

        np.array([0.0, 0.0, -3.0]),                    # ref_pos (desired hover at -3m Z)

        np.zeros(3),                                   # ref_vel
        np.zeros(3),                                   # ref_acc
        0.0,                                           # ref_yaw
        0.0,                                           # ref_yaw_rate
        "PX4_odometry_sample"                          # name
        ),

        (np.array([0.0, 0.0, 0.0]),
         np.array([0.0, 0.0, 0.0]),
         np.array([1.0, 0.0, 0.0, 0.0]),
         np.array([0.0, 0.0, 0.0]),
         np.array([0.0, 0.0, -3.0]),
         np.zeros(3),
         np.zeros(3),
         0.0, 0.0,
         "Hover at -3m"),

        # --- Already higher than target (should command down) ---
        (np.array([0.0, 0.0, -4.0]),
         np.array([0.0, 0.0, 0.0]),
         np.array([1.0, 0.0, 0.0, 0.0]),
         np.array([0.0, 0.0, 0.0]),
         np.array([0.0, 0.0, -3.0]),
         np.zeros(3),
         np.zeros(3),
         0.0, 0.0,
         "Climb from -4m to -3m"),

        # --- Already lower than target (should command up) ---
        (np.array([0.0, 0.0, -2.0]),
         np.array([0.0, 0.0, 0.0]),
         np.array([1.0, 0.0, 0.0, 0.0]),
         np.array([0.0, 0.0, 0.0]),
         np.array([0.0, 0.0, -3.0]),
         np.zeros(3),
         np.zeros(3),
         0.0, 0.0,
         "Descend from -2m to -3m"),

        # --- Offset in XY (should tilt to move) ---
        (np.array([2.0, -2.0, -3.0]),
         np.array([0.0, 0.0, 0.0]),
         np.array([1.0, 0.0, 0.0, 0.0]),
         np.array([0.0, 0.0, 0.0]),
         np.array([0.0, 0.0, -3.0]),
         np.zeros(3),
         np.zeros(3),
         0.0, 0.0,
         "XY correction"),

        # --- Moving reference velocity (should lead target) ---
        (np.array([0.0, 0.0, -3.0]),
         np.array([0.0, 0.0, 0.0]),
         np.array([1.0, 0.0, 0.0, 0.0]),
         np.array([0.0, 0.0, 0.0]),
         np.array([1.0, 1.0, -3.0]),
         np.array([0.5, 0.5, 0.0]),
         np.zeros(3),
         0.0, 0.0,
         "Track moving target"),

        # --- Hover with yaw command ---
        (np.array([0.0, 0.0, -3.0]),
         np.array([0.0, 0.0, 0.0]),
         np.array([1.0, 0.0, 0.0, 0.0]),
         np.array([0.0, 0.0, 0.0]),
         np.array([0.0, 0.0, -3.0]),
         np.zeros(3),
         np.zeros(3),
         np.pi/4, 0.0,
         "Hover at -3m with 45° yaw"),

        # --- Descend rapidly (high negative velocity) ---
        (np.array([0.0, 0.0, -1.0]),
        np.array([0.0, 0.0, -2.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, -3.0]),
        np.zeros(3),
        np.zeros(3),
        0.0, 0.0,
        "Fast descend to -3m"),

        # --- Move diagonally in XY while climbing ---
        (np.array([0.0, 0.0, -2.5]),
        np.array([0.1, 0.1, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, -2.0]),
        np.array([0.2, 0.2, 0.5]),
        np.zeros(3),
        np.pi/6, 0.0,
        "Diagonal climb to -2m with yaw"),

        # --- Hover with small position disturbance in XY ---
        (np.array([0.1, -0.1, -3.0]),
        np.zeros(3),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        np.array([0.0, 0.0, -3.0]),
        np.zeros(3),
        np.zeros(3),
        0.0, 0.0,
        "Hover at -3m with small XY offset"),

        # --- Rotate yaw while maintaining position ---
        (np.array([0.0, 0.0, -3.0]),
        np.zeros(3),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        np.array([0.0, 0.0, -3.0]),
        np.zeros(3),
        np.zeros(3),
        np.pi/2, 0.5,

        "Hover with yaw rotation at 90 deg and yaw rate 0.5 rad/s"),
        # --- Aggressive XY maneuver with velocity tracking ---
        (np.array([0.0, 0.0, -3.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        np.array([2.0, -2.0, -3.0]),
        np.array([1.0, -1.0, 0.0]),
        np.zeros(3),
        0.0, 0.0,
        "Fast XY maneuver to [2,-2,-3] with reference velocity"),

        # --- Climb to target with initial angular velocity ---
        (np.array([0.0, 0.0, -4.0]),
        np.zeros(3),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.1, -0.1, 0.2]),
        np.array([0.0, 0.0, -2.0]),
        np.zeros(3),
        np.zeros(3),
        0.0, 0.0,
        "Climb to -2m with initial angular velocity"),
    ]

    def check_physical_sense(normalized, throttles, desired_quat, mass=1.5, gravity=9.81, tol=1e-2):
        """
        Checks if the controller outputs make physical sense.
        Returns a dict with check results.
        """
        results = {}

        # 1. Normalized thrust should be within [0,1]
        results['normalized_range'] = np.all((normalized >= 0.0) & (normalized <= 1.0))

        # 2. Throttles should be within [0,1]
        results['throttles_range'] = np.all((throttles >= 0.0) & (throttles <= 1.0))

        # 3. Thrust must be at least enough to support weight (approximate)
        z_thrust = normalized[3] * mass * gravity
        results['thrust_positive'] = z_thrust >= 0.0
        results['thrust_sufficient'] = z_thrust >= 0.0  # simple check: always non-negative

        # 4. Quaternion should be normalized
        quat_norm = np.linalg.norm(desired_quat)
        results['quaternion_normalized'] = abs(quat_norm - 1.0) < tol

        # 5. Optional: warn if XY torques are too large (>10x weight*length, arbitrary)
        tau_xy_mag = np.linalg.norm(normalized[:2])
        results['xy_torque_reasonable'] = tau_xy_mag < 10.0

        return results
    
    # Run tests with automatic checks
    for (cur_pos, cur_vel, cur_q, cur_omega,
        ref_pos, ref_vel, ref_acc, ref_yaw, ref_yaw_rate, name) in test_cases:

        print(f"\n=== Test: {name} ===")
        normalized, throttles, desired_quat = controller.calculate_controller_output(
            kp, kv, kR, kw,
            cur_pos, cur_vel, cur_q, cur_omega,
            ref_pos, ref_vel, ref_acc, ref_yaw, ref_yaw_rate
        )
        print(f"Normalized: {np.round(normalized, 4)}")
        print(f"Throttles: {np.round(throttles, 4)}")
        print(f"Desired quaternion: {np.round(desired_quat, 4)}")

        # Check physical sense
        sense_check = check_physical_sense(normalized, throttles, desired_quat)
        for key, val in sense_check.items():
            status = "PASS" if val else "FAIL"
            print(f"{key}: {status}")


