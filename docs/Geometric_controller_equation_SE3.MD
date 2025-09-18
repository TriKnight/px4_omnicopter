# Geometric Controller Equations (SE(3) Quadrotor)
#### Paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5717652&tag=1

# Geometric Controller Equations (SE(3) Quadrotor)

## 1. Position & Velocity Errors
$$
e_p = p - p_r
$$
$$
e_v = v - v_r
$$

- $p \in \mathbb{R}^3$: current position of the UAV in the world frame (NED or ENU depending on convention)  
- $p_r \in \mathbb{R}^3$: reference (commanded) position  
- $v \in \mathbb{R}^3$: current velocity  
- $v_r \in \mathbb{R}^3$: reference velocity  

---

## 2. Desired Total Force
$$
f_d = -K_p e_p - K_v e_v + m g e_z + m a_r
$$

- $K_p = \mathrm{diag}(k_{px}, k_{py}, k_{pz})$: position gain matrix  
- $K_v = \mathrm{diag}(k_{vx}, k_{vy}, k_{vz})$: velocity gain matrix  
- $m$: UAV mass  
- $g$: gravitational acceleration ($\approx 9.81\ \mathrm{m/s^2}$)  
- $e_z = [0, 0, 1]^\top$: unit vector along the world $z$-axis (upward)  
- $a_r$: reference (feedforward) acceleration  

---

## 3. Thrust Along Body $z$-axis
$$
T = f_d^\top\, R_{BW}[:,3]
$$

- $T$: total thrust command (scalar)  
- $R_{BW} \in SO(3)$: rotation matrix body $\rightarrow$ world frame  
- $R_{BW}[:,3]$: third column of $R_{BW}$, i.e. the body $z$-axis expressed in world frame  

---

## 4. Desired Orientation

Desired body $z$-axis (align with desired force):
$$
b_{3d} = \frac{f_d}{\|f_d\|}
$$

Desired body $x$-axis (from reference yaw):
$$
b_{1c} =
\begin{bmatrix}
\cos(\psi_r) \\
\sin(\psi_r) \\
0
\end{bmatrix}
$$

Orthogonal body $y$-axis:
$$
b_{2d} = \frac{b_{3d} \times b_{1c}}{\|b_{3d} \times b_{1c}\|}
$$

Final desired rotation matrix:
$$
R_d =
\begin{bmatrix}
b_{2d} \times b_{3d} & b_{2d} & b_{3d}
\end{bmatrix}
$$

- $\psi_r$: reference yaw angle  
- $R_d \in SO(3)$: desired rotation matrix (orientation command for UAV)  

---

## 5. Attitude Error
$$
e_R = \tfrac{1}{2} \big( R_d^\top R - R^\top R_d \big)^\vee
$$

- $R$: current rotation matrix (from IMU/estimator)  
- $R_d$: desired rotation matrix  
- $(\cdot)^\vee$: vee operator (converts a skew-symmetric matrix to a vector)  

---

## 6. Angular Velocity Error
$$
e_\Omega = \Omega - R^\top R_d \Omega_r
$$

Reference angular velocity:
$$
\Omega_r = \dot{\psi_r}
\begin{bmatrix}
0 \\ 0 \\ 1
\end{bmatrix}
$$

- $\Omega$: current angular velocity (body frame)  
- $\Omega_r$: reference angular velocity (yaw rate)  

---

## 7. Control Torque
$$
\tau = -K_R e_R - K_\Omega e_\Omega + \Omega \times (J \Omega)
$$

- $K_R = \mathrm{diag}(k_{R_x}, k_{R_y}, k_{R_z})$: attitude gain matrix  
- $K_\Omega = \mathrm{diag}(k_{\omega_x}, k_{\omega_y}, k_{\omega_z})$: angular rate gain matrix  
- $J$: inertia matrix of the UAV (typically diagonal for small multirotors)  
- $\Omega \times (J \Omega)$: gyroscopic torque term  

---

## 8. Final Control Output
$$
u =
\begin{bmatrix}
\tau \\
T
\end{bmatrix}
$$

- $u$: control wrench (3 torques + thrust)  
- $\tau$: body torques  
- $T$: thrust command  

### ---------------------------------------------
```python

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

```