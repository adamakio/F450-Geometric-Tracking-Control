import numpy as np
from dataclasses import dataclass
from typing import Tuple


def hat(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def vee(S: np.ndarray) -> np.ndarray:
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def quat_to_rotmat_xyzw(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("Zero quaternion is invalid.")
    x, y, z, w = q / n
    return np.array([
        [1 - 2*(y**2 + z**2),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
    ])


def _normalize_with_derivatives(
    u: np.ndarray,
    u_dot: np.ndarray,
    u_ddot: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize u and propagate its first and second time derivatives.

    Returns (b, b_dot, b_ddot) where b = u / ||u||.
    Raises ValueError when ||u|| is below the degeneracy threshold.
    """
    norm = np.linalg.norm(u)
    if norm <= 1e-6:
        raise ValueError(f"Vector norm {norm} is below degeneracy threshold.")
    b = u / norm
    P = np.eye(3) - np.outer(b, b)
    b_dot = P @ u_dot / norm
    P_dot = -(np.outer(b_dot, b) + np.outer(b, b_dot))
    b_ddot = (
        (P_dot @ u_dot + P @ u_ddot) / norm
        - (u @ u_dot / norm**2) * b_dot
    )
    return b, b_dot, b_ddot


@dataclass
class ControlOutput:
    f: float
    M: np.ndarray
    normalized_throttle: np.ndarray
    e_x: np.ndarray
    e_v: np.ndarray
    b3_des: np.ndarray
    e_R: np.ndarray
    e_omega: np.ndarray
    omega_des: np.ndarray
    alpha_des: np.ndarray
    R_des: np.ndarray


class GeometricController:
    """Geometric controller mixin.

    Expects the following attributes to be set on self before calling
    compute_control():
      State:    x, v, q, omega, a, j
      Desired:  x_des, v_des, a_des, j_des, s_des,
                yaw_des, yaw_rate_des, yaw_accel_des
    """

    def __init__(
        self,
        k_x: float,
        k_v: float,
        K_R: np.ndarray,
        K_Omega: np.ndarray,
        m: float,
        g: float,
        J: np.ndarray,
        alloc_inv: np.ndarray,
    ):
        self.k_x = k_x
        self.k_v = k_v
        self.K_R = K_R
        self.K_Omega = K_Omega
        self.m = m
        self.mg = m * g
        self.J = J
        self.alloc_inv = alloc_inv

    def _compute_errors(self):
        self.e_x = self.x - self.x_des
        self.e_v = self.v - self.v_des

    def _compute_virtual_input(self):
        """Compute A = virtual force vector and its first and second derivatives."""
        e3 = np.array([0.0, 0.0, 1.0])
        self.A = -self.k_x * self.e_x - self.k_v * self.e_v + self.mg * e3 + self.m * self.a_des
        self.A_dot = - self.k_x * self.e_v - self.k_v * (self.a - self.a_des) + self.m * self.j_des 
        self.A_ddot =  - self.k_x * (self.a - self.a_des) - self.k_v * (self.j - self.j_des) + self.m * self.s_des

    def _compute_yaw_heading(self):
        """Compute the yaw heading vector c and its first and second derivatives."""
        psi = self.yaw_des
        psi_dot = self.yaw_rate_des
        psi_ddot = self.yaw_accel_des
        self.c = np.array([np.cos(psi), np.sin(psi), 0.0])
        self.c_dot = np.array([-np.sin(psi) * psi_dot, np.cos(psi) * psi_dot, 0.0])
        self.c_ddot = np.array([
            -np.cos(psi) * psi_dot**2 - np.sin(psi) * psi_ddot,
            -np.sin(psi) * psi_dot**2 + np.cos(psi) * psi_ddot,
            0.0,
        ])

    def _compute_desired_thrust_axis(self):
        """Compute b_{3d} = A/||A|| and its derivatives. Falls back to e3."""
        e3 = np.array([0.0, 0.0, 1.0])
        try:
            self.b_3d, self.b_3d_dot, self.b_3d_ddot = _normalize_with_derivatives(
                self.A, self.A_dot, self.A_ddot
            )
        except ValueError:
            self.b_3d = e3.copy()
            self.b_3d_dot = np.zeros(3)
            self.b_3d_ddot = np.zeros(3)

    def _compute_desired_lateral_axis(self):
        """Compute b_{2d} = (b_{3d} x c)/||...|| and its derivatives.

        Falls back to the yaw-perpendicular direction when b_{3d} and c are parallel.
        """
        r = np.cross(self.b_3d, self.c)
        r_dot = np.cross(self.b_3d_dot, self.c) + np.cross(self.b_3d, self.c_dot)
        r_ddot = (
            np.cross(self.b_3d_ddot, self.c)
            + 2.0 * np.cross(self.b_3d_dot, self.c_dot)
            + np.cross(self.b_3d, self.c_ddot)
        )
        try:
            self.b_2d, self.b_2d_dot, self.b_2d_ddot = _normalize_with_derivatives(r, r_dot, r_ddot)
        except ValueError:
            psi = self.yaw_des
            self.b_2d = np.array([-np.sin(psi), np.cos(psi), 0.0])
            self.b_2d_dot = np.zeros(3)
            self.b_2d_ddot = np.zeros(3)

    def _compute_desired_forward_axis(self):
        """Compute b_{1d} = b_{2d} x b_{3d} and its derivatives."""
        self.b_1d = np.cross(self.b_2d, self.b_3d)
        self.b_1d_dot = np.cross(self.b_2d_dot, self.b_3d) + np.cross(self.b_2d, self.b_3d_dot)
        self.b_1d_ddot = (
            np.cross(self.b_2d_ddot, self.b_3d)
            + 2.0 * np.cross(self.b_2d_dot, self.b_3d_dot)
            + np.cross(self.b_2d, self.b_3d_ddot)
        )

    def _compute_desired_frame(self):
        """Assemble R_des from the three desired frame axes."""
        self.R_des = np.column_stack((self.b_1d, self.b_2d, self.b_3d))

    def _compute_desired_rates(self):
        """Compute the desired body-frame angular velocity and acceleration."""
        self.omega_des = np.array([
            np.dot(self.b_3d, self.b_2d_dot),
            np.dot(self.b_1d, self.b_3d_dot),
            np.dot(self.b_2d, self.b_1d_dot),
        ])
        self.alpha_des = np.array([
            np.dot(self.b_3d_dot, self.b_2d_dot) + np.dot(self.b_3d, self.b_2d_ddot),
            np.dot(self.b_1d_dot, self.b_3d_dot) + np.dot(self.b_1d, self.b_3d_ddot),
            np.dot(self.b_2d_dot, self.b_1d_dot) + np.dot(self.b_2d, self.b_1d_ddot),
        ])

    def _compute_attitude_errors(self):
        """Compute the attitude error e_R and the angular rate error e_omega."""
        self.R = quat_to_rotmat_xyzw(self.q)
        self.e_R = 0.5 * vee(self.R_des.T @ self.R - self.R.T @ self.R_des)
        self.e_omega = self.omega - self.R.T @ self.R_des @ self.omega_des 

    def _compute_thrust_moment(self):
        """Compute the collective thrust f and the moment vector M."""
        e3 = np.array([0.0, 0.0, 1.0])
        self.f = np.dot(self.A, self.R @ e3)
        self.M = (
            -self.K_R @ self.e_R - self.K_Omega @ self.e_omega
            + np.cross(self.omega, self.J @ self.omega)
            - self.J @ (
                hat(self.omega) @ self.R.T @ self.R_des @ self.omega_des
                - self.R.T @ self.R_des @ self.alpha_des
            )
        )

    def _compute_throttle(self):
        """Map thrust and moment to per-rotor normalized throttle commands."""
        self.normalized_throttle = self.alloc_inv @ np.array([self.f, self.M[0], self.M[1], self.M[2]])

    def compute_control(self) -> ControlOutput:
        self._compute_errors()
        self._compute_virtual_input()
        self._compute_yaw_heading()
        self._compute_desired_thrust_axis()
        self._compute_desired_lateral_axis()
        self._compute_desired_forward_axis()
        self._compute_desired_frame()
        self._compute_desired_rates()
        self._compute_attitude_errors()
        self._compute_thrust_moment()
        self._compute_throttle()
        return ControlOutput(
            f=self.f,
            M=self.M,
            normalized_throttle=self.normalized_throttle,
            e_x=self.e_x,
            e_v=self.e_v,
            b3_des=self.b_3d,
            e_R=self.e_R,
            e_omega=self.e_omega,
            omega_des=self.omega_des,
            alpha_des=self.alpha_des,
            R_des=self.R_des,
        )
