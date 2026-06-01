import numpy as np


def _quat_to_rotmat_xyzw(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y**2 + z**2),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
    ])


def _quat_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """q_dot = 0.5 * Ξ(q) @ omega,  q = [qx, qy, qz, qw]."""
    qx, qy, qz, qw = q
    xi = np.array([
        [ qw, -qz,  qy],
        [ qz,  qw, -qx],
        [-qy,  qx,  qw],
        [-qx, -qy, -qz],
    ])
    return 0.5 * xi @ omega


class QuadrotorDynamics:
    """Quadrotor rigid-body dynamics mixin with RK4 integration.

    State vector layout (length 13):
        [0:3]  position       x        (world frame, m)
        [3:6]  velocity       v        (world frame, m/s)
        [6:10] quaternion     q        (xyzw convention)
        [10:13] angular vel   omega    (body frame, rad/s)

    Control input:
        throttle[i] ∈ [0, 1]  — normalised throttle for rotor i.
        Force from rotor i: F_i = k_f * throttle_i  (N).
        Wrench = allocation_matrix @ throttle.
    """

    def __init__(
        self,
        m: float,
        g: float,
        J: np.ndarray,
        alloc_matrix: np.ndarray,
    ):
        self._m = m
        self._g = g
        self._J = J
        self._J_inv = np.linalg.inv(J)
        self._alloc = alloc_matrix

    # ---------------------------------------------------------------- dynamics

    def _state_derivative(self, state: np.ndarray, throttle: np.ndarray) -> np.ndarray:
        x     = state[0:3]   # noqa: F841
        v     = state[3:6]
        q     = state[6:10] / np.linalg.norm(state[6:10])
        omega = state[10:13]

        R  = _quat_to_rotmat_xyzw(q)
        e3 = np.array([0.0, 0.0, 1.0])

        wrench = self._alloc @ np.clip(throttle, 0.0, 1.0)
        f      = wrench[0]
        M      = wrench[1:4]

        x_dot     = v
        v_dot     = (f / self._m) * (R @ e3) - self._g * e3
        q_dot     = _quat_derivative(q, omega)
        omega_dot = self._J_inv @ (M - np.cross(omega, self._J @ omega))

        return np.concatenate([x_dot, v_dot, q_dot, omega_dot])

    def rk4_step(self, state: np.ndarray, throttle: np.ndarray, dt: float) -> np.ndarray:
        """Advance state by dt using a classic fourth-order Runge-Kutta step."""
        k1 = self._state_derivative(state,              throttle)
        k2 = self._state_derivative(state + 0.5*dt*k1, throttle)
        k3 = self._state_derivative(state + 0.5*dt*k2, throttle)
        k4 = self._state_derivative(state +     dt*k3, throttle)

        next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        next_state[6:10] /= np.linalg.norm(next_state[6:10])
        return next_state

    def euler_step(self, state: np.ndarray, throttle: np.ndarray, dt: float) -> np.ndarray:
        """Advance state by dt using a first-order Euler step."""
        next_state = state + dt * self._state_derivative(state, throttle)

        next_state[6:10] /= np.linalg.norm(next_state[6:10])
        return next_state

    def acceleration(self, state: np.ndarray, throttle: np.ndarray) -> np.ndarray:
        """Return translational acceleration (world frame) for the given state and input."""
        deriv = self._state_derivative(state, throttle)
        return deriv[3:6]
