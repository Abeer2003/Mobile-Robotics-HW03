import numpy as np

from ekf import Ekf, wrap


class EkfSlam(Ekf):
    NP = 5

    def __init__(self, x0, P0, params):
        super().__init__(x0, P0, params)
        self.n = 0
        self.cands = []
        self.cand_radius = self.p.get('cand_radius', 0.25)
        self.cand_count = self.p.get('cand_count', 3)
        self.cand_ttl = self.p.get('cand_ttl', 1.0)
        self.gate_new = self.p.get('gate_new', 30.0)

    def predict(self, ds, dtheta, omega=None):
        u = np.array([ds, dtheta])
        F, _ = self.jacobians(self.x, u)
        Qk = self.Q(self.x, u)
        self.x[:5] = self.f(self.x[:5], u)
        self.x[2] = wrap(self.x[2])
        P = self.P
        P[:5, :5] = F @ P[:5, :5] @ F.T + Qk
        if P.shape[0] > 5:
            P[:5, 5:] = F @ P[:5, 5:]
            P[5:, :5] = P[:5, 5:].T

    def _H_gyro(self):
        H = np.zeros((1, len(self.x)))
        H[0, 3] = H[0, 4] = 1.0
        return H

    def update_zero_rate(self, omega):
        h = np.array([self.x[3] + self.x[4]])
        R = np.array([[self.R_scale * self.p['sigma_n'] ** 2]])
        return self._update(np.array([omega]), h, self._H_gyro(), R)

    def update_gyro(self, omega, ds, dtheta):
        h = np.array([dtheta / self.dt + self.x[3] + self.x[4]])
        R = np.array([[self.R_scale * (self.p['sigma_n'] ** 2
                                       + self.p['k2'] * abs(dtheta) / self.dt ** 2)]])
        return self._update(np.array([omega]), h, self._H_gyro(), R)

    def lm(self, j):
        return self.x[5 + 2 * j:7 + 2 * j]

    def R_lm(self):
        return np.diag([self.p['sigma_r'] ** 2, self.p['sigma_b'] ** 2])

    def H_lm(self, j):
        Hp = self.landmark_H(self.x, self.lm(j))
        H = Hp.copy()
        H[:, 5 + 2 * j:7 + 2 * j] = -Hp[:, 0:2]
        return H

    def d2_map(self, j, r, beta):
        h, _, _ = self.landmark_h(self.x, self.lm(j))
        H = self.H_lm(j)
        nu = np.array([r, beta]) - h
        nu[1] = wrap(nu[1])
        S = H @ self.P @ H.T + self.R_lm()
        return float(nu @ np.linalg.solve(S, nu))

    def associate_map(self, r, beta, gate, exclude=()):
        best, best_d2 = None, np.inf
        for j in range(self.n):
            if j in exclude:
                continue
            d2 = self.d2_map(j, r, beta)
            if d2 < best_d2:
                best, best_d2 = j, d2
        if best_d2 > gate:
            return None, best_d2
        return best, best_d2

    def update_landmark_idx(self, j, r, beta):
        h, _, _ = self.landmark_h(self.x, self.lm(j))
        return self._update(np.array([r, beta]), h, self.H_lm(j), self.R_lm(), angle_rows=(1,))

    def init_landmark(self, pose, r, beta):
        c, s = np.cos(pose[2]), np.sin(pose[2])
        wx = self.p['cam_x'] + r * np.cos(beta)
        wy = self.p['cam_y'] + r * np.sin(beta)
        g = np.array([pose[0] + c * wx - s * wy, pose[1] + s * wx + c * wy])
        Jx = np.array([[1.0, 0.0, -(g[1] - pose[1])],
                       [0.0, 1.0, g[0] - pose[0]]])
        Rot = np.array([[c, -s], [s, c]])
        Jz = Rot @ np.array([[np.cos(beta), -r * np.sin(beta)],
                             [np.sin(beta), r * np.cos(beta)]])
        return g, Jx, Jz

    def add_landmark(self, r, beta):
        g, Jx3, Jz = self.init_landmark(self.x[:3], r, beta)
        N = len(self.x)
        Jx = np.zeros((2, N))
        Jx[:, :3] = Jx3
        Pn = np.zeros((N + 2, N + 2))
        Pn[:N, :N] = self.P
        Pn[:N, N:] = self.P @ Jx.T
        Pn[N:, :N] = Jx @ self.P
        Pn[N:, N:] = Jx @ self.P @ Jx.T + Jz @ self.R_lm() @ Jz.T
        self.x = np.concatenate([self.x, g])
        self.P = 0.5 * (Pn + Pn.T)
        self.n += 1
        return self.n - 1

    def process_detection(self, r, beta, t, used, gate):
        j, d2 = self.associate_map(r, beta, gate, used)
        if j is not None:
            nis = self.update_landmark_idx(j, r, beta)
            used.add(j)
            return 'update', j, nis
        if d2 <= self.gate_new:
            return 'ambiguous', None, None
        self.cands = [c for c in self.cands if t - c['t'] < self.cand_ttl]
        g, _, _ = self.init_landmark(self.x[:3], r, beta)
        for c in self.cands:
            if np.hypot(*(c['pos'] - g)) < self.cand_radius:
                c['n'] += 1
                c['pos'] = g
                c['t'] = t
                if c['n'] >= self.cand_count:
                    self.cands = [q for q in self.cands if q is not c]
                    j = self.add_landmark(r, beta)
                    used.add(j)
                    return 'new', j, None
                return 'cand', None, None
        self.cands.append(dict(pos=g, n=1, t=t))
        return 'cand', None, None
