
import numpy as np

from ekf import Ekf

PARAMS = dict(sigma_n=0.015, sigma_g=0.0042, tau=3.4, k1=1e-3, k2=1e-3, dt=0.01,
              cam_x=0.20, cam_y=0.08, sigma_r=0.02, sigma_b=0.01)


def make():
    return Ekf([0.3, -0.2, 0.7, -0.01, 0.002], np.eye(5) * 1e-3, PARAMS)


def num_jac(fun, v, eps=1e-6):
    v = np.asarray(v, float)
    cols = []
    for i in range(len(v)):
        d = np.zeros_like(v); d[i] = eps
        cols.append((fun(v + d) - fun(v - d)) / (2 * eps))
    return np.array(cols).T


def test_check_F():
    e, u = make(), np.array([0.008, 0.012])
    F, _ = e.jacobians(e.x, u)
    assert np.allclose(F, num_jac(lambda x: e.f(x, u), e.x), atol=1e-8)


def test_check_G():
    e, u = make(), np.array([0.008, 0.012])
    _, G = e.jacobians(e.x, u)
    assert np.allclose(G, num_jac(lambda uu: e.f(e.x, uu), u), atol=1e-8)


def test_check_H():
    e = make()
    for h in (lambda x: np.array([x[3] + x[4]]),
              lambda x: np.array([0.012 / e.dt + x[3] + x[4]])):
        assert np.allclose(Ekf.H_GYRO, num_jac(h, e.x), atol=1e-8)


def test_Q_symmetric_psd():
    e = make()
    Q = e.Q(e.x, np.array([0.008, 0.012]))
    assert np.allclose(Q, Q.T) and np.linalg.eigvalsh(Q).min() > -1e-15


def test_associate_matches_nis():
    e = make()
    lms = {1: np.array([2.5, -5.92]), 2: np.array([-3.0, 4.0]), 3: np.array([5.0, 1.0])}
    h, _, _ = e.landmark_h(e.x, lms[1])
    r, b = h[0] + 0.01, h[1] - 0.005
    j, d2 = e.associate(r, b, lms, gate=9.21)
    assert j == 1
    e2 = make()
    assert np.isclose(d2, e2.update_landmark(lms[1], r, b))
    assert e.associate(r + 5.0, b, lms, gate=9.21)[0] is None


def test_check_H_landmark():
    e, m = make(), np.array([2.5, -5.92])
    H = e.landmark_H(e.x, m)
    num = num_jac(lambda x: e.landmark_h(x, m)[0], e.x)
    assert np.allclose(H, num, atol=1e-7)
