"""
State x = [x, y, theta, b0, g]      Input u = [ds, dtheta]
"""
import numpy as np


def wrap(a):
    """Wrap an angle to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class Ekf:
    def __init__(self, x0, P0, params):
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)
        self.p = dict(params)
        self.dt = self.p['dt']
        self.phi = np.exp(-self.dt / self.p['tau'])
        self.sigma_g_q = self.p.get('sigma_g_q', self.p['sigma_g'])
        self.R_scale = self.p.get('R_scale', 1.0)

    # ---- 4(a) motion model ------------------------------------------------------
    def f(self, x, u):
        ds, dth = u
        A = x[2] + 0.5 * dth
        return np.array([x[0] + ds * np.cos(A),
                         x[1] + ds * np.sin(A),
                         x[2] + dth,
                         x[3],
                         self.phi * x[4]])

    def jacobians(self, x, u):
        ds, dth = u
        A = x[2] + 0.5 * dth
        c, s = np.cos(A), np.sin(A)
        F = np.array([[1, 0, -ds * s, 0, 0],
                      [0, 1, ds * c, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, self.phi]])
        G = np.array([[c, -0.5 * ds * s],
                      [s, 0.5 * ds * c],
                      [0, 1],
                      [0, 0],
                      [0, 0]])
        return F, G

    def Q(self, x, u):
        _, G = self.jacobians(x, u)
        Sigma = np.diag([self.p['k1'] * abs(u[0]), self.p['k2'] * abs(u[1])])
        Qb = np.zeros((5, 5))
        Qb[4, 4] = self.sigma_g_q ** 2 * (1 - self.phi ** 2)
        return G @ Sigma @ G.T + Qb

    def predict(self, ds, dtheta, omega=None):
        """x = f(x, u) with theta wrapped, P = F P F^T + Q (F, G at the old state)."""
        u = np.array([ds, dtheta])
        F, _ = self.jacobians(self.x, u)
        Qk = self.Q(self.x, u)
        self.x = self.f(self.x, u)
        self.x[2] = wrap(self.x[2])
        self.P = F @ self.P @ F.T + Qk

    # ---- 4(b) gyro measurements -------------------------------------------------
    H_GYRO = np.array([[0.0, 0.0, 0.0, 1.0, 1.0]])

    def update_zero_rate(self, omega):
        """Stationary: h = b0 + g, R = sigma_n^2. Returns the NIS."""
        h = np.array([self.x[3] + self.x[4]])
        R = np.array([[self.R_scale * self.p['sigma_n'] ** 2]])
        return self._update(np.array([omega]), h, self.H_GYRO, R)

    def update_gyro(self, omega, ds, dtheta):
        """Moving: h = dtheta/dt + b0 + g, R = sigma_n^2 + k2|dtheta|/dt^2. Returns the NIS."""
        h = np.array([dtheta / self.dt + self.x[3] + self.x[4]])
        R = np.array([[self.R_scale * (self.p['sigma_n'] ** 2
                                       + self.p['k2'] * abs(dtheta) / self.dt ** 2)]])
        return self._update(np.array([omega]), h, self.H_GYRO, R)

    def landmark_h(self, x, m_xy):
        c, s = np.cos(x[2]), np.sin(x[2])
        dx, dy = m_xy[0] - x[0], m_xy[1] - x[1]
        qx = c * dx + s * dy - self.p['cam_x']
        qy = -s * dx + c * dy - self.p['cam_y']
        return np.array([np.hypot(qx, qy), np.arctan2(qy, qx)]), qx, qy

    def landmark_H(self, x, m_xy):
        c, s = np.cos(x[2]), np.sin(x[2])
        _, qx, qy = self.landmark_h(x, m_xy)
        cx, cy = self.p['cam_x'], self.p['cam_y']
        r2 = qx ** 2 + qy ** 2
        r = np.sqrt(r2)
        H = np.zeros((2, len(x)))
        H[0, 0] = (-qx * c + qy * s) / r
        H[0, 1] = (-qx * s - qy * c) / r
        H[0, 2] = (qx * cy - qy * cx) / r
        H[1, 0] = (qx * s + qy * c) / r2
        H[1, 1] = (qy * s - qx * c) / r2
        H[1, 2] = -1.0 - (qx * cx + qy * cy) / r2
        return H

    def update_landmark(self, m_xy, r, beta):
        h, _, _ = self.landmark_h(self.x, m_xy)
        H = self.landmark_H(self.x, m_xy)
        R = np.diag([self.p['sigma_r'] ** 2, self.p['sigma_b'] ** 2])
        return self._update(np.array([r, beta]), h, H, R, angle_rows=(1,))

    def landmark_d2(self, m_xy, r, beta):
        h, _, _ = self.landmark_h(self.x, m_xy)
        H = self.landmark_H(self.x, m_xy)
        R = np.diag([self.p['sigma_r'] ** 2, self.p['sigma_b'] ** 2])
        nu = np.array([r, beta]) - h
        nu[1] = wrap(nu[1])
        S = H @ self.P @ H.T + R
        return float(nu @ np.linalg.solve(S, nu))

    def associate(self, r, beta, landmarks, gate):
        best, best_d2 = None, np.inf
        for j, m_xy in landmarks.items():
            d2 = self.landmark_d2(m_xy, r, beta)
            if d2 < best_d2:
                best, best_d2 = j, d2
        if best_d2 > gate:
            return None, best_d2
        return best, best_d2

    # ---- generic EKF update (Joseph form) ---------------------------------------
    def _update(self, z, h, H, R, angle_rows=()):
        nu = np.atleast_1d(z - h).astype(float)
        for i in angle_rows:
            nu[i] = wrap(nu[i])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ nu
        self.x[2] = wrap(self.x[2])
        IKH = np.eye(len(self.x)) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return float(nu @ np.linalg.solve(S, nu))
