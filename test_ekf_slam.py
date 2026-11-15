import numpy as np

from ekf_slam import EkfSlam

PARAMS = dict(sigma_n=0.015, sigma_g=0.0042, tau=3.4, k1=1e-3, k2=1e-3, dt=0.01,
              cam_x=0.20, cam_y=0.08, sigma_r=0.02, sigma_b=0.01)


def num_jac(fun, v, eps=1e-6):
    v = np.asarray(v, float)
    cols = []
    for i in range(len(v)):
        d = np.zeros_like(v)
        d[i] = eps
        cols.append((fun(v + d) - fun(v - d)) / (2 * eps))
    return np.array(cols).T


def make():
    e = EkfSlam([0.3, -0.2, 0.7, -0.01, 0.002], np.eye(5) * 1e-3, PARAMS)
    e.add_landmark(3.0, 0.2)
    e.add_landmark(4.0, -0.3)
    return e


def test_init_jacobians():
    e = EkfSlam([0.3, -0.2, 0.7, -0.01, 0.002], np.eye(5) * 1e-3, PARAMS)
    r, b = 3.0, 0.2
    _, Jx, Jz = e.init_landmark(e.x[:3], r, b)
    assert np.allclose(Jx, num_jac(lambda p: e.init_landmark(p, r, b)[0], e.x[:3]), atol=1e-6)
    assert np.allclose(Jz, num_jac(lambda z: e.init_landmark(e.x[:3], z[0], z[1])[0], np.array([r, b])), atol=1e-6)


def test_new_landmark_reproduces_reading():
    e = EkfSlam([0.3, -0.2, 0.7, -0.01, 0.002], np.eye(5) * 1e-3, PARAMS)
    e.add_landmark(3.0, 0.2)
    h, _, _ = e.landmark_h(e.x, e.lm(0))
    assert np.allclose(h, [3.0, 0.2], atol=1e-9)


def test_covariance_symmetric_psd():
    e = make()
    assert np.allclose(e.P, e.P.T)
    assert np.linalg.eigvalsh(e.P).min() > -1e-12


def test_H_lm_columns():
    e = make()
    for j in range(2):
        H = e.H_lm(j)
        num = num_jac(lambda x: e.landmark_h(x[:5], x[5 + 2 * j:7 + 2 * j])[0], e.x)
        assert np.allclose(H, num, atol=1e-6)


def test_predict_keeps_landmarks():
    e = make()
    m0 = e.x[5:].copy()
    Pmm = e.P[5:, 5:].copy()
    e.predict(0.01, 0.005)
    assert np.allclose(e.x[5:], m0) and np.allclose(e.P[5:, 5:], Pmm)
    assert np.allclose(e.P, e.P.T)


def test_d2_equals_nis():
    e = make()
    h, _, _ = e.landmark_h(e.x, e.lm(0))
    r, b = h[0] + 0.01, h[1] - 0.004
    d2 = e.d2_map(0, r, b)
    assert np.isclose(d2, e.update_landmark_idx(0, r, b))
