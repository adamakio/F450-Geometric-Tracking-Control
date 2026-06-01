#!/usr/bin/env python3
"""
Standalone simulation to verify the geometric tracking controller before
deploying in IsaacSim / ROS 2.

Reference:
  Lee, Leok, McClamroch — "Geometric Tracking Control of a Quadrotor UAV
  on SE(3)", CDC 2010.

Body-frame convention (differs from the paper):
  • b3 points UP  (paper has b3 pointing DOWN)
  • Thrust acts along +Re3  (paper: -Re3)
  • Inertial z-axis is UP, gravity = -g·e3
  • Equations of motion:  mv̇ = -mg·e3 + f·Re3

The controller algebra is equivalent — the sign difference in A is absorbed:
  A_code  = -k_x·e_x - k_v·e_v + mg·e3 + m·a_des   (= -A_paper)
  b3_des  =  A_code / ‖A_code‖                       (= b3d in paper)
  f       =  dot(A_code, Re3)                         (= eq. 15 in paper)
  M       =  same as eq. 16 in paper

Allocation matrix — X-configuration (rotors at 45° from body axes):
  [f ]   [  1      1      1      1  ] [f1]
  [Mx] = [-d/√2  d/√2  d/√2  -d/√2 ] [f2]
  [My]   [-d/√2  d/√2 -d/√2   d/√2 ] [f3]
  [Mz]   [-k_m  -k_m   k_m    k_m  ] [f4]

Run:
  conda run -n geometric_control_poc python simulate_geometric_control.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import logm


# ── Math utilities ─────────────────────────────────────────────────────────────

def hat(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric (hat) map: R^3 → so(3)."""
    return np.array([
        [0,     -v[2],  v[1]],
        [v[2],   0,    -v[0]],
        [-v[1],  v[0],  0   ],
    ])


def vee(S: np.ndarray) -> np.ndarray:
    """Inverse hat (vee) map: so(3) → R^3."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def project_SO3(R: np.ndarray) -> np.ndarray:
    """Project a near-rotation matrix back to SO(3) via SVD."""
    U, _, Vt = np.linalg.svd(R)
    R_proj = U @ Vt
    if np.linalg.det(R_proj) < 0:
        U[:, -1] *= -1
        R_proj = U @ Vt
    return R_proj


def attitude_error_psi(R: np.ndarray, R_des: np.ndarray) -> float:
    """Ψ(R, Rd) = ½ tr(I − Rd^T R)  (Lee 2010, Eq. 8).
    Ψ ∈ [0, 2];  Ψ < 1 ↔ error < 90°;  Ψ = 2 ↔ 180°."""
    return 0.5 * np.trace(np.eye(3) - R_des.T @ R)


# ── Geometric controller ───────────────────────────────────────────────────────

class GeometricController:
    """
    Implements the geometric tracking controller from Lee 2010,
    adapted for z-up / b3-up body frame convention.
    """

    def __init__(self, m: float, g: float, J: np.ndarray,
                 k_x: float, k_v: float, k_R: float, k_Omega: float):
        self.m  = m
        self.g  = g
        self.mg = m * g
        self.J  = J
        self.k_x     = k_x
        self.k_v     = k_v
        self.k_R     = k_R
        self.k_Omega = k_Omega
        self._e3 = np.array([0., 0., 1.])

    def compute(
        self,
        x: np.ndarray, v: np.ndarray,
        R: np.ndarray, Omega: np.ndarray,
        x_des: np.ndarray, v_des: np.ndarray,
        a_des: np.ndarray, yaw_des: float,
        R_des_prev: np.ndarray, omega_des_prev: np.ndarray,
        dt: float,
    ):
        """
        Compute control force f (N) and moment M (N·m).

        Parameters
        ----------
        dt : pass a negative value on the first call (initialises Ωd = 0).

        Returns
        -------
        f, M, R_des, omega_des, diagnostics
        """
        e3 = self._e3

        # Position / velocity errors  (Lee 2010, Eq. 6-7)
        e_x = x - x_des
        e_v = v - v_des

        # Desired body-z axis  (Lee 2010, Eq. 12, sign-adapted for z-up)
        A = -self.k_x * e_x - self.k_v * e_v + self.mg * e3 + self.m * a_des
        A_norm = np.linalg.norm(A)
        b3_des = A / A_norm if A_norm > 1e-6 else e3.copy()

        # Desired attitude from yaw + b3_des  (Lee 2010, Sec. III-B)
        b1_yaw = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.])
        b2_des = np.cross(b3_des, b1_yaw)
        b2_norm = np.linalg.norm(b2_des)
        if b2_norm > 1e-6:
            b2_des /= b2_norm
        else:
            b2_des = np.array([-np.sin(yaw_des), np.cos(yaw_des), 0.])
        b1_des = np.cross(b2_des, b3_des)
        R_des = np.column_stack((b1_des, b2_des, b3_des))

        # Desired angular velocity / acceleration via matrix log finite-diff
        if dt > 1e-6:
            S = np.real(logm(R_des_prev.T @ R_des)) / dt
            omega_des = vee(S)
            alpha_des = (omega_des - omega_des_prev) / dt
        else:
            omega_des = np.zeros(3)
            alpha_des = np.zeros(3)

        # Attitude and angular-rate errors  (Lee 2010, Eq. 10-11)
        e_R     = 0.5 * vee(R_des.T @ R - R.T @ R_des)
        e_Omega = Omega - R.T @ R_des @ omega_des

        # Control force and moment  (Lee 2010, Eq. 15-16)
        f = np.dot(A, R @ e3)
        M = (
            -self.k_R * e_R
            - self.k_Omega * e_Omega
            + np.cross(Omega, self.J @ Omega)
            - self.J @ (hat(Omega) @ R.T @ R_des @ omega_des
                        - R.T @ R_des @ alpha_des)
        )

        diagnostics = dict(
            e_x=e_x, e_v=e_v, e_R=e_R, e_Omega=e_Omega,
            b3_des=b3_des, A=A,
        )
        return f, M, R_des, omega_des, diagnostics


# ── Quadrotor dynamics ─────────────────────────────────────────────────────────

class QuadrotorDynamics:
    """
    Rigid-body equations of motion + X-configuration allocation matrix.

    Allocation  [f, Mx, My, Mz]^T = alloc @ [f1, f2, f3, f4]^T
    """

    def __init__(self, m: float, g: float, J: np.ndarray,
                 d: float, k_m: float):
        self.m   = m
        self.g   = g
        self.J   = J
        sq2 = np.sqrt(2)
        self.alloc = np.array([
            [1,        1,        1,        1       ],
            [-d/sq2,   d/sq2,    d/sq2,   -d/sq2  ],
            [-d/sq2,   d/sq2,   -d/sq2,    d/sq2  ],
            [-k_m,    -k_m,      k_m,      k_m    ],
        ])
        self.alloc_inv = np.linalg.inv(self.alloc)

    def fM_to_rotors(self, f: float, M: np.ndarray) -> np.ndarray:
        """Map total force + moments to individual rotor thrust forces (N)."""
        return self.alloc_inv @ np.array([f, M[0], M[1], M[2]])

    def deriv(self, x, v, R, Omega, f, M):
        """Equations of motion (z-up, thrust = +fRe3)."""
        e3 = np.array([0., 0., 1.])
        x_dot     = v
        v_dot     = -self.g * e3 + (f / self.m) * (R @ e3)
        R_dot     = R @ hat(Omega)
        Omega_dot = np.linalg.solve(self.J, M - np.cross(Omega, self.J @ Omega))
        return x_dot, v_dot, R_dot, Omega_dot

    def step_rk4(self, x, v, R, Omega, f, M, dt):
        """RK4 integration with mid-step SO(3) projection."""
        def k(x_, v_, R_, O_):
            return self.deriv(x_, v_, R_, O_, f, M)

        k1 = k(x, v, R, Omega)
        k2 = k(x + .5*dt*k1[0], v + .5*dt*k1[1],
                project_SO3(R + .5*dt*k1[2]), Omega + .5*dt*k1[3])
        k3 = k(x + .5*dt*k2[0], v + .5*dt*k2[1],
                project_SO3(R + .5*dt*k2[2]), Omega + .5*dt*k2[3])
        k4 = k(x + dt*k3[0], v + dt*k3[1],
                project_SO3(R + dt*k3[2]), Omega + dt*k3[3])

        c = dt / 6.0
        x_new = x + c * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        v_new = v + c * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        R_new = project_SO3(R + c * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]))
        O_new = Omega + c * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
        return x_new, v_new, R_new, O_new


# ── Simulation runner ──────────────────────────────────────────────────────────

def simulate(controller: GeometricController,
             dynamics: QuadrotorDynamics,
             trajectory_fn,          # callable: t → dict(x, v, a, yaw)
             t_end: float,
             dt: float = 0.004,      # 250 Hz — matches the ROS 2 node default
             x0=None, v0=None, R0=None, Omega0=None) -> dict:
    """
    Integrate the closed-loop system and return a dict of logged arrays.

    trajectory_fn(t) must return a dict with keys:
        x   : (3,) desired position (m)
        v   : (3,) desired velocity (m/s)
        a   : (3,) desired acceleration (m/s²)
        yaw : float, desired yaw (rad)
    """
    x     = np.zeros(3)  if x0     is None else np.array(x0, float)
    v     = np.zeros(3)  if v0     is None else np.array(v0, float)
    R     = np.eye(3)    if R0     is None else np.array(R0, float)
    Omega = np.zeros(3)  if Omega0 is None else np.array(Omega0, float)

    R_des_prev     = np.eye(3)
    omega_des_prev = np.zeros(3)

    n = int(round(t_end / dt)) + 1
    keys = ['t', 'x', 'v', 'x_des', 'v_des',
            'e_x', 'e_v', 'e_R', 'e_Omega',
            'Psi', 'f', 'M', 'rotors', 'R_det']
    logs = {k: [] for k in keys}

    for i in range(n):
        t    = i * dt
        traj = trajectory_fn(t)

        ctrl_dt = dt if i > 0 else -1.0   # sentinel → ωd = αd = 0 on step 0
        f, M, R_des, omega_des, diag = controller.compute(
            x, v, R, Omega,
            traj['x'], traj['v'], traj['a'], traj['yaw'],
            R_des_prev, omega_des_prev, ctrl_dt,
        )

        rotor_f = dynamics.fM_to_rotors(f, M)

        logs['t'].append(t)
        logs['x'].append(x.copy())
        logs['v'].append(v.copy())
        logs['x_des'].append(traj['x'].copy())
        logs['v_des'].append(traj['v'].copy())
        logs['e_x'].append(diag['e_x'].copy())
        logs['e_v'].append(diag['e_v'].copy())
        logs['e_R'].append(diag['e_R'].copy())
        logs['e_Omega'].append(diag['e_Omega'].copy())
        logs['Psi'].append(attitude_error_psi(R, R_des))
        logs['f'].append(f)
        logs['M'].append(M.copy())
        logs['rotors'].append(rotor_f.copy())
        logs['R_det'].append(np.linalg.det(R))

        x, v, R, Omega = dynamics.step_rk4(x, v, R, Omega, f, M, dt)
        R_des_prev     = R_des
        omega_des_prev = omega_des

    for k in logs:
        logs[k] = np.array(logs[k])
    return logs


# ── Trajectory generators ──────────────────────────────────────────────────────

def traj_hover(pos=(0., 0., 1.), yaw=0.):
    pos = np.asarray(pos, float)
    def fn(t):
        return dict(x=pos.copy(), v=np.zeros(3), a=np.zeros(3), yaw=yaw)
    return fn


def traj_step(target=(2., 0., 1.), t_step=0., yaw=0.):
    """Step to target at t = t_step; hover at origin before that."""
    target = np.asarray(target, float)
    def fn(t):
        pos = target if t >= t_step else np.zeros(3)
        return dict(x=pos.copy(), v=np.zeros(3), a=np.zeros(3), yaw=yaw)
    return fn


def traj_circle(radius=1.0, freq=0.3, height=1.5, yaw=0.):
    """Circle in the xy-plane with velocity and acceleration feedforward."""
    w = 2 * np.pi * freq
    def fn(t):
        x = np.array([radius*np.cos(w*t), radius*np.sin(w*t), height])
        v = np.array([-radius*w*np.sin(w*t), radius*w*np.cos(w*t), 0.])
        a = np.array([-radius*w**2*np.cos(w*t), -radius*w**2*np.sin(w*t), 0.])
        return dict(x=x, v=v, a=a, yaw=yaw)
    return fn


def traj_elliptic_helix():
    """
    Case I from Lee 2010: elliptic helix with rotating heading.
      xd(t) = [0.4t,  0.4·sin(πt),  0.6·cos(πt)]
      b1d(t) = [cos(πt), sin(πt), 0]   ↔  yaw = πt
    """
    def fn(t):
        x = np.array([0.4*t,
                       0.4*np.sin(np.pi*t),
                       0.6*np.cos(np.pi*t)])
        v = np.array([0.4,
                       0.4*np.pi*np.cos(np.pi*t),
                      -0.6*np.pi*np.sin(np.pi*t)])
        a = np.array([0.,
                      -0.4*np.pi**2*np.sin(np.pi*t),
                      -0.6*np.pi**2*np.cos(np.pi*t)])
        return dict(x=x, v=v, a=a, yaw=np.pi*t)
    return fn


def traj_hover_at_origin():
    """Hover at origin — used for attitude-recovery test."""
    def fn(t):
        return dict(x=np.zeros(3), v=np.zeros(3), a=np.zeros(3), yaw=0.)
    return fn


# ── Health check ───────────────────────────────────────────────────────────────

def check_health(logs: dict, label: str = "") -> bool:
    """Print a health summary; return True if all checks pass."""
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")

    ok = True

    # Non-finite values
    for key in ['x', 'v', 'e_R', 'e_Omega', 'f', 'M']:
        if not np.all(np.isfinite(logs[key])):
            print(f"  [FAIL] Non-finite values in '{key}'")
            ok = False
    if ok:
        print("  [PASS] All signals are finite")

    # SO(3) integrity  (det R should stay ≈ 1)
    det_err = np.abs(logs['R_det'] - 1.0)
    if det_err.max() > 1e-4:
        print(f"  [WARN] det(R) deviated: max error = {det_err.max():.2e}")
    else:
        print(f"  [PASS] SO(3) integrity: max |det(R)-1| = {det_err.max():.2e}")

    # Final position error
    e_x_final = np.linalg.norm(logs['e_x'][-1])
    Psi_final = logs['Psi'][-1]
    print(f"  Final position error ‖ex‖ = {e_x_final:.4f} m")
    print(f"  Final attitude Ψ           = {Psi_final:.4f}  (< 0.01 is good)")

    # Negative rotor forces (infeasible without bi-directional rotors)
    neg = (logs['rotors'] < -0.01).sum()
    if neg > 0:
        worst = logs['rotors'].min()
        print(f"  [WARN] {neg} samples with negative rotor thrust "
              f"(min = {worst:.2f} N) — expected during large-angle recovery")
    else:
        print(f"  [PASS] All rotor forces ≥ 0  (max = {logs['rotors'].max():.2f} N)")

    return ok


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_results(logs: dict, title: str = ""):
    t = logs['t']
    xyz = ['x', 'y', 'z']
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    fig, axes = plt.subplots(3, 3, figsize=(17, 11))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # 1 — Position tracking
    ax = axes[0, 0]
    for i, lb in enumerate(xyz):
        ax.plot(t, logs['x'][:, i],     color=colors[i], label=f'{lb}')
        ax.plot(t, logs['x_des'][:, i], color=colors[i], linestyle='--',
                alpha=0.6, label=f'{lb}_des')
    ax.set_title('Position (m)')
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlabel('t (s)')
    ax.grid(True)

    # 2 — Position error
    ax = axes[0, 1]
    for i, lb in enumerate(xyz):
        ax.plot(t, logs['e_x'][:, i], color=colors[i], label=f'e_{lb}')
    ax.plot(t, np.linalg.norm(logs['e_x'], axis=1), 'k--', label='‖ex‖')
    ax.set_title('Position error (m)')
    ax.legend(fontsize=7)
    ax.set_xlabel('t (s)')
    ax.grid(True)

    # 3 — Velocity error
    ax = axes[0, 2]
    for i, lb in enumerate(xyz):
        ax.plot(t, logs['e_v'][:, i], color=colors[i], label=f'e_v{lb}')
    ax.set_title('Velocity error (m/s)')
    ax.legend(fontsize=7)
    ax.set_xlabel('t (s)')
    ax.grid(True)

    # 4 — Attitude error function Ψ
    ax = axes[1, 0]
    ax.plot(t, logs['Psi'], 'tab:purple', linewidth=1.5)
    ax.axhline(1.0, color='r',  linestyle='--', label='Ψ=1  (90°)')
    ax.axhline(2.0, color='k',  linestyle='--', label='Ψ=2 (180°)')
    ax.set_ylim(-0.05, max(2.1, logs['Psi'].max() * 1.05))
    ax.set_title('Attitude error Ψ  (Lee 2010, Eq. 8)')
    ax.legend(fontsize=8)
    ax.set_xlabel('t (s)')
    ax.grid(True)

    # 5 — Rotation error e_R
    ax = axes[1, 1]
    for i, lb in enumerate(xyz):
        ax.plot(t, logs['e_R'][:, i], color=colors[i], label=f'eR_{lb}')
    ax.set_title('Attitude error e_R (rad)')
    ax.legend(fontsize=7)
    ax.set_xlabel('t (s)')
    ax.grid(True)

    # 6 — Angular rate error e_Ω
    ax = axes[1, 2]
    for i, lb in enumerate(xyz):
        ax.plot(t, logs['e_Omega'][:, i], color=colors[i], label=f'eΩ_{lb}')
    ax.set_title('Angular-rate error e_Ω (rad/s)')
    ax.legend(fontsize=7)
    ax.set_xlabel('t (s)')
    ax.grid(True)

    # 7 — Total thrust f
    ax = axes[2, 0]
    ax.plot(t, logs['f'], color='tab:brown')
    ax.axhline(0., color='k', linestyle='--', alpha=0.4)
    mg = 1.806 * 9.8065
    ax.axhline(mg, color='gray', linestyle=':', label=f'mg = {mg:.1f} N')
    ax.set_title('Total thrust f (N)')
    ax.legend(fontsize=8)
    ax.set_xlabel('t (s)')
    ax.grid(True)

    # 8 — Moments M
    ax = axes[2, 1]
    for i, lb in enumerate(['Mx', 'My', 'Mz']):
        ax.plot(t, logs['M'][:, i], color=colors[i], label=lb)
    ax.set_title('Control moment M (N·m)')
    ax.legend(fontsize=7)
    ax.set_xlabel('t (s)')
    ax.grid(True)

    # 9 — Rotor forces
    ax = axes[2, 2]
    rotor_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for i in range(4):
        ax.plot(t, logs['rotors'][:, i], color=rotor_colors[i], label=f'f{i+1}')
    ax.axhline(0., color='k', linestyle='--', alpha=0.5, label='0 N (min)')
    ax.set_title('Rotor forces (N)')
    ax.legend(fontsize=7)
    ax.set_xlabel('t (s)')
    ax.grid(True)

    plt.tight_layout()
    return fig


def plot_3d(logs: dict, title: str = ""):
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot(logs['x'][:, 0],     logs['x'][:, 1],     logs['x'][:, 2],
            linewidth=2, label='actual')
    ax.plot(logs['x_des'][:, 0], logs['x_des'][:, 1], logs['x_des'][:, 2],
            '--', linewidth=1.5, alpha=0.7, label='desired')
    ax.scatter(*logs['x'][0],  color='green', s=60, zorder=5, label='start')
    ax.scatter(*logs['x'][-1], color='red',   s=60, zorder=5, label='end')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
    ax.set_title(title)
    ax.legend()
    return fig


# ── Default system parameters ──────────────────────────────────────────────────

def build_system():
    """Construct controller + dynamics with the node's default parameters."""
    m    = 1.806
    g    = 9.8065
    J    = np.diag([0.016, 0.017, 0.024])
    d    = 0.215
    k_m  = 8e-4

    ctrl = GeometricController(
        m=m, g=g, J=J,
        k_x=16.0, k_v=5.6, k_R=8.81, k_Omega=2.54,
    )
    dyn = QuadrotorDynamics(m=m, g=g, J=J, d=d, k_m=k_m)
    return ctrl, dyn


# ── Test cases ─────────────────────────────────────────────────────────────────

def run_all_tests(save_figs: bool = False):
    ctrl, dyn = build_system()
    dt = 0.004   # 250 Hz — matches the ROS 2 node's publish_rate

    figs = []

    # ── Test 1: Hover ──────────────────────────────────────────────────────────
    print("\nRunning Test 1: Hover at [0, 0, 1] m …")
    logs = simulate(ctrl, dyn, traj_hover([0., 0., 1.]), t_end=5.0, dt=dt)
    check_health(logs, "Test 1 — Hover at [0, 0, 1] m")
    figs.append(plot_results(logs, "Test 1 — Hover at [0, 0, 1] m"))
    figs.append(plot_3d(logs,      "Test 1 — Hover (3-D)"))

    # ── Test 2: Position step ──────────────────────────────────────────────────
    print("\nRunning Test 2: Step from origin to [2, 1, 1.5] m …")
    logs = simulate(ctrl, dyn,
                    traj_step(target=[2., 1., 1.5], t_step=0.),
                    t_end=10.0, dt=dt)
    check_health(logs, "Test 2 — Position step to [2, 1, 1.5] m")
    figs.append(plot_results(logs, "Test 2 — Position step to [2, 1, 1.5] m"))
    figs.append(plot_3d(logs,      "Test 2 — Position step (3-D)"))

    # ── Test 3: Circular trajectory ────────────────────────────────────────────
    print("\nRunning Test 3: Circular trajectory (r=1 m, f=0.3 Hz) …")
    logs = simulate(ctrl, dyn,
                    traj_circle(radius=1.0, freq=0.3, height=1.5),
                    t_end=10.0, dt=dt,
                    x0=[1., 0., 1.5])   # start on the circle
    check_health(logs, "Test 3 — Circular trajectory")
    figs.append(plot_results(logs, "Test 3 — Circular trajectory"))
    figs.append(plot_3d(logs,      "Test 3 — Circular trajectory (3-D)"))

    # ── Test 4: Elliptic helix  (Lee 2010, Case I) ─────────────────────────────
    print("\nRunning Test 4: Elliptic helix (Lee 2010 Case I) …")
    logs = simulate(ctrl, dyn,
                    traj_elliptic_helix(),
                    t_end=10.0, dt=dt)
    check_health(logs, "Test 4 — Elliptic helix (Lee 2010 Case I)")
    figs.append(plot_results(logs, "Test 4 — Elliptic helix (Lee 2010 Case I)"))
    figs.append(plot_3d(logs,      "Test 4 — Elliptic helix (3-D)"))

    # ── Test 5: Recovery from near-upside-down  (Lee 2010, Case II) ────────────
    # Ψ(R0, I) ≈ 1.995  →  almost-global attractiveness regime (Prop. 3)
    print("\nRunning Test 5: Recovery from near-upside-down (Lee 2010 Case II) …")
    R0_paper = np.array([
        [1,  0,       0      ],
        [0, -0.9995, -0.0314 ],
        [0,  0.0314, -0.9995 ],
    ])
    logs = simulate(ctrl, dyn,
                    traj_hover_at_origin(),
                    t_end=8.0, dt=dt,
                    R0=R0_paper)
    check_health(logs, "Test 5 — Recovery from near-upside-down (Lee 2010 Case II)")
    figs.append(plot_results(logs, "Test 5 — Recovery from near-upside-down (Lee 2010 Case II)"))

    if save_figs:
        import os
        out_dir = "sim_figures"
        os.makedirs(out_dir, exist_ok=True)
        names = [
            "test1_hover", "test1_hover_3d",
            "test2_step",  "test2_step_3d",
            "test3_circle","test3_circle_3d",
            "test4_helix", "test4_helix_3d",
            "test5_flip",
        ]
        for fig, name in zip(figs, names):
            path = os.path.join(out_dir, f"{name}.png")
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {path}")

    plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Simulate geometric controller for quadrotor UAV.')
    parser.add_argument('--save', action='store_true',
                        help='Save figures to sim_figures/ instead of displaying')
    args = parser.parse_args()
    run_all_tests(save_figs=args.save)
