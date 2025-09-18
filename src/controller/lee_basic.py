import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# --- Utilities ---
def hat(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def vee(M):
    return np.array([M[2,1], M[0,2], M[1,0]])

# --- Controller ---
class GeometricController:
    def __init__(self, mass=1.0, inertia=np.diag([0.01, 0.01, 0.02]),
                 kx=8.0, kv=3.0, kR=4.0, kOmega=0.1, g=9.81):
        self.m = mass
        self.J = inertia
        self.kx = kx
        self.kv = kv
        self.kR = kR
        self.kOmega = kOmega
        self.g = g
        self.e3 = np.array([0,0,1])
        
    def compute_control(self, x, v, R_mat, Omega, xd, vd, a_d, Rd=None, Omega_d=None):
        ex = x - xd
        ev = v - vd
        F_des = -self.kx*ex - self.kv*ev + self.m*(a_d + self.g*self.e3)

        b3 = R_mat[:,2]
        f = np.dot(F_des, b3)

        if Rd is None:
            b3_des = F_des / np.linalg.norm(F_des)
            b1_des = np.array([1,0,0])
            b1_des = b1_des - np.dot(b1_des, b3_des)*b3_des
            b1_des /= np.linalg.norm(b1_des)
            b2_des = np.cross(b3_des, b1_des)
            Rd = np.column_stack((b1_des, b2_des, b3_des))

        eR = 0.5 * vee(Rd.T @ R_mat - R_mat.T @ Rd)
        eOmega = Omega - (Omega_d if Omega_d is not None else np.zeros(3))

        M = -self.kR*eR - self.kOmega*eOmega + np.cross(Omega, self.J @ Omega)
        return f, M

# --- Simulation ---
if __name__ == "__main__":
    dt = 0.01
    T = 5.0
    steps = int(T/dt)

    ctrl = GeometricController()

    # Initial state
    x = np.array([0.0, 0.0, 0.0])   # position
    v = np.zeros(3)                 # velocity
    R_mat = np.eye(3)                # attitude
    Omega = np.zeros(3)              # angular velocity

    # Desired trajectory: hover at (0,0,1)
    xd = np.array([0.0, 0.0, -2.0])
    vd = np.zeros(3)
    a_d = np.zeros(3)

    # Logs
    pos_log = []
    thrust_log = []
    moment_log = []

    def expm_SO3(omega):
        theta = np.linalg.norm(omega)
        if theta < 1e-6:
            return np.eye(3)
        k = omega/theta
        K = hat(k)
        return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)

    for k in range(steps):
        f, M = ctrl.compute_control(x, v, R_mat, Omega, xd, vd, a_d)

        # --- Physics integration (simple Euler integration) ---
        acc = (f * R_mat @ np.array([0,0,1])) / ctrl.m - np.array([0,0,ctrl.g])
        v += acc * dt
        x += v * dt

        # Angular dynamics
        Omega_dot = np.linalg.inv(ctrl.J) @ (M - np.cross(Omega, ctrl.J @ Omega))
        Omega += Omega_dot * dt

        # Update rotation
        R_mat = R_mat @ expm_SO3(Omega * dt)  # integrate rotation

        # Logging
        pos_log.append(x.copy())
        thrust_log.append(f)
        moment_log.append(M.copy())

    pos_log = np.array(pos_log)
    thrust_log = np.array(thrust_log)
    moment_log = np.array(moment_log)

    # --- Plot results ---
    t = np.linspace(0, T, steps)

    plt.figure(figsize=(10,6))
    plt.subplot(3,1,1)
    plt.plot(t, pos_log[:,2], label="z")
    plt.axhline(-2.0, color='r', linestyle='--', label="z desired")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.grid()

    plt.subplot(3,1,2)
    plt.plot(t, thrust_log, label="Thrust")
    plt.ylabel("Thrust (N)")
    plt.legend()
    plt.grid()

    plt.subplot(3,1,3)
    plt.plot(t, moment_log[:,0], label="Mx")
    plt.plot(t, moment_log[:,1], label="My")
    plt.plot(t, moment_log[:,2], label="Mz")
    plt.ylabel("Moments (Nm)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
