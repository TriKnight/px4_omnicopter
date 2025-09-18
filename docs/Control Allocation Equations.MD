# Control Allocation Equations

We want to map the **desired wrench**

$$
w =
\begin{bmatrix}
\tau_x \\ \tau_y \\ \tau_z \\ T
\end{bmatrix}
\in \mathbb{R}^4
$$

(where $\tau_x, \tau_y, \tau_z$ are body torques and $T$ is thrust)  

into **rotor speeds, throttles, and normalized outputs**.

---

### 1. Rotor Speeds (squared)
The allocation matrix $A \in \mathbb{R}^{N \times 4}$ (depends on number of arms, geometry, thrust/moment constants):

$$
\omega^2 = A \, w
$$

- $\omega^2 \in \mathbb{R}^N$: squared rotor speeds  
- $w \in \mathbb{R}^4$: desired wrench  
- $A = \text{torques\_and\_thrust\_to\_rotor\_velocities}$  

If any element of $\omega^2$ is negative, Python version clips it to 0:

$$
\omega^2_i \gets \max(0, \omega^2_i)
$$

---

### 2. Rotor Speeds
Take square root elementwise:

$$
\omega = \sqrt{\omega^2}, \quad \omega \in \mathbb{R}^N
$$

---

### 3. Throttle Conversion
Throttle is computed from rotor speed with an offset $\omega_0$ and scaling factor $s$:

$$
u_i = \frac{\omega_i - \omega_0}{s}, \quad i=1,\dots,N
$$

or in vector form:

$$
u = \frac{\omega - \omega_0 \mathbf{1}_N}{s}
$$

- $u \in \mathbb{R}^N$: throttle commands  
- $\omega_0 = \text{zero\_position\_armed}$  
- $s = \text{input\_scaling}$  
- $\mathbf{1}_N$: vector of ones  

---

### 4. Normalized Torques & Thrust
PX4 expects a normalized 4-vector:

$$
\tilde{w} = B \, u
$$

where  

- $\tilde{w} \in \mathbb{R}^4$: normalized $[\tau_x, \tau_y, \tau_z, T]$  
- $B = \text{throttles\_to\_normalized\_torques\_and\_thrust}$  

---

### ✅ Final Mapping
Putting all steps together:

$$
\tilde{w} = B \, \frac{\sqrt{\max(0, A w)} - \omega_0 \mathbf{1}_N}{s}
$$

$$
u = \frac{\sqrt{\max(0, A w)} - \omega_0 \mathbf{1}_N}{s}
$$

## CODE 
-------------------------
```python
    # ------------------ control allocation (inverse) -------------------------
def px4_inverse_sitl(self, wrench):
    omega_sq = self.torques_thrust_to_rotor_velocities.dot(wrench)
    omega_sq = np.clip(omega_sq, 0.0, None)
    omega = np.sqrt(omega_sq)
    throttles = (omega - (self.zero_position_armed * np.ones_like(omega))) / max(1e-6, self.input_scaling)
    normalized = self.throttles_to_normalized.dot(throttles)
    return normalized, throttles
```
The function `px4_inverse_sitl`  converts a desired *wrench* (control wrench = body torques + total thrust),
$$
\mathbf{\omega} = \begin{bmatrix}\tau_x \\ \tau_y \\ \tau_z \\ T\end{bmatrix} \in \mathbb{R}^4,
$$
into per-rotor commands that PX4 / SITL can apply. Concretely it computes:

1. the *squared rotor speeds* (or a rotor-speed-like quantity) that would produce that wrench (via a pseudo-inverse of the mixing matrix),
2. the rotor speeds $\omega$ by element-wise square root,
3. the *throttle* command for each rotor (offset + scaling),
4. a final linear transform from those throttles to a "normalized torque/thrust" vector used elsewhere.

This is the control-allocation / inverse mixing step: *wrench → rotor speeds → throttles → normalized outputs*.

---




## Step-by-step equations

**1) Mixing matrix (geometry × physical scale).**  
Start from geometry matrix (for quad-X) $G_0 \in \mathbb{R}^{4\times n}$, e.g. for quad (rows: roll, pitch, yaw, thrust):
$$
G_0 = \begin{bmatrix}
- \sin(45^\circ) & \sin(45^\circ) & \sin(45^\circ) & -\sin(45^\circ) \\
- \sin(45^\circ) & \sin(45^\circ) & -\sin(45^\circ) & \sin(45^\circ) \\
-1 & -1 & 1 & 1 \\
1 & 1 & 1 & 1
\end{bmatrix}.
$$

Scale rows by physical constants (diagonal $K$):
$$
K = \mathrm{diag}\big( \kappa_\text{thrust}\,\ell,\; \kappa_\text{thrust}\,\ell,\; \kappa_\text{moment}\,\kappa_\text{thrust},\; \kappa_\text{thrust} \big)
$$

Then the full mapping from rotor-speed-squared vector $\mathbf{s}$ to wrench $\mathbf{w}$ is:
$$
\mathbf{w} \;=\; G\,\mathbf{s}, \qquad G \;=\; K\,G_0.
$$

Interpretation: each rotor contributes to roll/pitch as $\ell\cdot \kappa_\text{thrust}\cdot \omega^2$ terms, to yaw via moment constant, and to thrust via thrust constant.

---

**2) Pseudoinverse (wrench → rotor-speed-squared)**  
Compute pseudoinverse of $G$ (numerically: COD pseudo inverse or Moore–Penrose):
$$
G^\dagger \;=\; G^{+} \in \mathbb{R}^{n\times 4}.
$$

Then the (nominal) rotor-speed-squared vector solving $G \mathbf{s} = \mathbf{w}$ is:
$$
\mathbf{s} \;=\; G^\dagger \,\mathbf{w}.
$$

(If $G$ is square and invertible, $G^\dagger = G^{-1}$; for real systems pseudoinverse handles redundancy/noise.)

---

**3) Non-negativity & elementwise sqrt to rotor speeds**  
Rotor speeds cannot be negative; code clamps then takes sqrt:

$$
\mathbf{s} \leftarrow \max( \mathbf{s},\; \mathbf{0} ) \quad\text{(elementwise clamp)},
$$

$$
\boldsymbol{\omega} \;=\; \sqrt{\mathbf{s}} \qquad (\text{elementwise}).
$$

This yields rotor angular speeds (rad/s) or a speed-like proxy.

---

**4) Convert rotor speeds to throttle command**  
The C++ code then computes *throttles* as:

$$
\mathbf{t} \;=\; \frac{\boldsymbol{\omega} - \omega_0 \mathbf{1}}{s}
$$

where $\omega_0$ is `_zero_position_armed` (small idle offset), $s$ is `_input_scaling` (scale factor), and division is elementwise. In python:

throttles = (omega - zero_position_armed * ones) / input_scaling

```python
# ------------------ matrix setup -----------------------------
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
```