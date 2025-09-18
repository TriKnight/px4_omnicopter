#!/usr/bin/env python3

import numpy as np
from math import sin, cos, sqrt, atan2
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
# QoS for PX4 topics: BEST_EFFORT
qos_profile = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)
from scipy.spatial.transform import Rotation as R

def quat_to_rotmat(q):
    """Convert quaternion [x, y, z, w] to rotation matrix."""
    return R.from_quat(q).as_matrix()


def quaternion_from_rotmat(rotmat):
    """Convert rotation matrix to quaternion [x, y, z, w]."""
    return R.from_matrix(rotmat).as_quat()

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
# def quat_to_rotmat(q):
#     """Convert quaternion [w,x,y,z] to rotation matrix (3x3)."""
#     w, x, y, z = q
#     R = np.array([
#         [1 - 2*(y*y+z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
#         [2*(x*y + z*w),   1 - 2*(x*x+z*z),       2*(y*z - x*w)],
#         [2*(x*z - y*w),       2*(y*z + x*w), 1 - 2*(x*x+y*y)]
#     ])
#     return R

# def quaternion_from_rotmat(R):
#     """Convert rotation matrix to quaternion [w,x,y,z]."""
#     trace = np.trace(R)
#     if trace > 0:
#         s = 0.5 / sqrt(trace + 1.0)
#         w = 0.25 / s
#         x = (R[2,1] - R[1,2]) * s
#         y = (R[0,2] - R[2,0]) * s
#         z = (R[1,0] - R[0,1]) * s
#     else:
#         if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
#             s = 2.0 * sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
#             w = (R[2,1] - R[1,2]) / s
#             x = 0.25 * s
#             y = (R[0,1] + R[1,0]) / s
#             z = (R[0,2] + R[2,0]) / s
#         elif R[1,1] > R[2,2]:
#             s = 2.0 * sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
#             w = (R[0,2] - R[2,0]) / s
#             x = (R[0,1] + R[1,0]) / s
#             y = 0.25 * s
#             z = (R[1,2] + R[2,1]) / s
#         else:
#             s = 2.0 * sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
#             w = (R[1,0] - R[0,1]) / s
#             x = (R[0,2] + R[2,0]) / s
#             y = (R[1,2] + R[2,1]) / s
#             z = 0.25 * s
#     return np.array([w, x, y, z])

def rotvec_FLD_to_FRD(v):
    """Rotate vector from FLU to FRD frame (flip y/z)."""
    return np.array([v[0], -v[1], -v[2]])
