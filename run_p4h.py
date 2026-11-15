import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Ellipse
from scipy.stats import chi2

from ekf import Ekf, wrap

PAIR = 'P06'
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', '..', 'data', PAIR)
OUT = os.path.join(HERE, '..', 'plots')
STOP_EPS = 1e-9

run = pd.read_csv(os.path.join(DATA, 'run.csv'))
det = pd.read_csv(os.path.join(DATA, 'detections.csv'))
gp = yaml.safe_load(open(os.path.join(DATA, 'gyro_params.yaml')))
info = yaml.safe_load(open(os.path.join(DATA, 'run_info.yaml')))
lm = yaml.safe_load(open(os.path.join(HERE, '..', '..', 'config', 'landmarks.yaml')))['landmarks']
MAP = {d['id']: np.array([d['x'], d['y']]) for d in lm}
g60 = pd.read_csv(os.path.join(DATA, 'gyro_60s.csv')).gyro_z.values

PARAMS = dict(sigma_n=gp['sigma_n'], sigma_g=gp['sigma_g'], tau=gp['tau'],
              k1=info['odometry_noise']['k1'], k2=info['odometry_noise']['k2'], dt=gp['dt'],
              cam_x=0.20, cam_y=0.08, sigma_r=0.02, sigma_b=0.01)
x0 = [info['initial_pose']['x'], info['initial_pose']['y'], info['initial_pose']['theta'],
      float(g60.mean()), 0.0]
P0 = np.diag([1e-6, 1e-6, 1e-6, gp['sigma_g'] ** 2, gp['sigma_g'] ** 2])

t = run.time.values
ds, dth, om = run.ds.values, run.dtheta.values, run.gyro_z.values
gt = run[['gt_x', 'gt_y', 'gt_theta']].values
n = len(run)
still = (np.abs(ds) < STOP_EPS) & (np.abs(dth) < STOP_EPS)

step = np.rint(det.time.values / PARAMS['dt']).astype(int) - 1
by_step = {}
for i, k in enumerate(step):
    by_step.setdefault(k, []).append(i)

ekf = Ekf(x0, P0, PARAMS)
X, P = np.zeros((n, 5)), np.zeros((n, 5, 5))
lm_t, lm_nis, lm_id = [], [], []
for k in range(n):
    ekf.predict(ds[k], dth[k], om[k])
    if still[k]:
        ekf.update_zero_rate(om[k])
    else:
        ekf.update_gyro(om[k], ds[k], dth[k])
    for i in by_step.get(k, []):
        d = det.iloc[i]
        lm_nis.append(ekf.update_landmark(MAP[int(d.id)], d.range, d.bearing))
        lm_t.append(t[k])
        lm_id.append(int(d.id))
    X[k], P[k] = ekf.x, ekf.P

lm_t, lm_nis, lm_id = np.array(lm_t), np.array(lm_nis), np.array(lm_id)
np.savez(os.path.join(OUT, 'p4_partII_result.npz'), t=t, X=X, P=P, lm_t=lm_t, lm_nis=lm_nis)

r1 = np.load(os.path.join(OUT, 'p4_partI_result.npz'))
X1, P1 = r1['X'], r1['P']


def ellipses(ax, Xe, Pe, color, scale=1.0):
    for k in range(0, n, 500):
        w, v = np.linalg.eigh(Pe[k, :2, :2])
        ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
        ax.add_patch(Ellipse(Xe[k, :2], scale * 4 * np.sqrt(max(w[1], 0)), scale * 4 * np.sqrt(max(w[0], 0)),
                             angle=ang, fill=False, color=color, lw=0.8))


fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
for ax, Xe, Pe, title, sc in [(axs[0], X1, P1, 'Part I: wheels + gyro (2$\\sigma$ ellipses every 5 s)', 1.0),
                              (axs[1], X, P, 'Part II: + landmarks (2$\\sigma$ ellipses every 5 s, enlarged x10)', 10.0)]:
    ax.plot(gt[:, 0], gt[:, 1], 'g-', lw=3, label='ground truth')
    ax.plot(Xe[:, 0], Xe[:, 1], 'r--', lw=1, label='EKF')
    ellipses(ax, Xe, Pe, 'b', sc)
    ax.set_aspect('equal'); ax.grid(True); ax.set_title(title)
    ax.set_xlabel('x (m)')
for m in MAP.values():
    axs[1].plot(m[0], m[1], 'k^', ms=5)
axs[0].set_xlim(gt[:, 0].min() - 1.5, gt[:, 0].max() + 1.5)
axs[0].set_ylim(gt[:, 1].min() - 2, gt[:, 1].max() + 1)
axs[0].set_ylabel('y (m)'); axs[0].legend(loc='upper right')
fig.savefig(os.path.join(OUT, 'p4h_path.png'), dpi=150, bbox_inches='tight')

fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
for i, (a, lab) in enumerate(zip(axs, ['x (m)', 'y (m)', '$\\theta$ (rad)'])):
    tru = gt[:, i]
    e1, e2 = X1[:, i], X[:, i]
    if i == 2:
        tru = np.unwrap(tru)
        e1 = np.unwrap(e1); e1 = e1 + 2 * np.pi * np.round((tru[0] - e1[0]) / (2 * np.pi))
        e2 = np.unwrap(e2); e2 = e2 + 2 * np.pi * np.round((tru[0] - e2[0]) / (2 * np.pi))
    s1, s2 = np.sqrt(P1[:, i, i]), np.sqrt(P[:, i, i])
    a.plot(t, e1, 'r', label='Part I'); a.fill_between(t, e1 - 2 * s1, e1 + 2 * s1, color='r', alpha=0.2)
    a.plot(t, e2, 'b', label='Part II'); a.fill_between(t, e2 - 2 * s2, e2 + 2 * s2, color='b', alpha=0.3)
    a.plot(t, tru, 'g', label='ground truth')
    a.set_ylabel(lab); a.grid(True)
axs[0].legend(ncol=3); axs[-1].set_xlabel('time (s)')
fig.savefig(os.path.join(OUT, 'p4h_pose.png'), dpi=150, bbox_inches='tight')

fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
for i, (a, lab) in enumerate(zip(axs, ['x error (m)', 'y error (m)', '$\\theta$ error (rad, wrapped)'])):
    f = (lambda X_: X_[:, i] - gt[:, i]) if i < 2 else (lambda X_: wrap(X_[:, 2] - gt[:, 2]))
    s1, s2 = np.sqrt(P1[:, i, i]), np.sqrt(P[:, i, i])
    a.plot(t, f(X1), 'r', label='Part I'); a.fill_between(t, -2 * s1, 2 * s1, color='r', alpha=0.2)
    a.plot(t, f(X), 'b', label='Part II'); a.fill_between(t, -2 * s2, 2 * s2, color='b', alpha=0.3)
    a.set_ylabel(lab); a.grid(True)
axs[0].legend(ncol=2); axs[-1].set_xlabel('time (s)')
fig.savefig(os.path.join(OUT, 'p4h_pose_error.png'), dpi=150, bbox_inches='tight')

def bias_terms(Xe, Pe):
    return [(Xe[:, 3], Pe[:, 3, 3]), (Xe[:, 4], Pe[:, 4, 4]),
            (Xe[:, 3] + Xe[:, 4], Pe[:, 3, 3] + Pe[:, 4, 4] + 2 * Pe[:, 3, 4])]


fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
for a, lab, (y1, v1), (y2, v2) in zip(axs, ['$\\hat b_0$ (rad/s)', '$\\hat g_k$ (rad/s)', '$\\hat b_0+\\hat g_k$ (rad/s)'],
                                      bias_terms(X1, P1), bias_terms(X, P)):
    s1, s2 = np.sqrt(v1), np.sqrt(v2)
    a.plot(t, y1, 'r', label='Part I'); a.fill_between(t, y1 - 2 * s1, y1 + 2 * s1, color='r', alpha=0.2)
    a.plot(t, y2, 'b', label='Part II'); a.fill_between(t, y2 - 2 * s2, y2 + 2 * s2, color='b', alpha=0.3)
    a.set_ylabel(lab); a.grid(True)
axs[0].legend(ncol=2); axs[-1].set_xlabel('time (s)')
fig.savefig(os.path.join(OUT, 'p4h_bias.png'), dpi=150, bbox_inches='tight')

N = len(lm_nis)
lo1, hi1 = chi2.ppf([0.025, 0.975], 2)
W = 20
ma = np.convolve(lm_nis, np.ones(W) / W, mode='valid')
loW, hiW = chi2.ppf([0.025, 0.975], 2 * W) / W
loN, hiN = chi2.ppf([0.025, 0.975], 2 * N) / N
fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
axs[0].semilogy(lm_t, np.maximum(lm_nis, 1e-6), '.', ms=2)
axs[0].axhline(lo1, color='r', ls='--'); axs[0].axhline(hi1, color='r', ls='--')
axs[0].set_ylabel('NIS (2 DOF)'); axs[0].set_title('Landmark updates: per-step NIS, $\\chi^2_2$ 95% bounds')
axs[1].plot(lm_t[W - 1:], ma)
axs[1].axhline(loW, color='r', ls='--'); axs[1].axhline(hiW, color='r', ls='--'); axs[1].axhline(2.0, color='k', ls=':')
axs[1].set_ylabel('20-update moving average'); axs[1].set_xlabel('time (s)')
for a in axs: a.grid(True)
fig.savefig(os.path.join(OUT, 'p4h_nis.png'), dpi=150, bbox_inches='tight')

e1 = np.hypot(X1[:, 0] - gt[:, 0], X1[:, 1] - gt[:, 1])
e2 = np.hypot(X[:, 0] - gt[:, 0], X[:, 1] - gt[:, 1])
print(f'landmark updates N={N}, time-averaged NIS = {lm_nis.mean():.3f}, 95% bounds for the mean [{loN:.3f}, {hiN:.3f}]')
print(f'per-step inside [{lo1:.3f}, {hi1:.3f}]: {np.mean((lm_nis >= lo1) & (lm_nis <= hi1)) * 100:.1f}%, '
      f'MA inside [{loW:.3f}, {hiW:.3f}]: {np.mean((ma >= loW) & (ma <= hiW)) * 100:.1f}%')
print(f'position RMSE: Part I {np.sqrt(np.mean(e1 ** 2)):.4f} m, Part II {np.sqrt(np.mean(e2 ** 2)):.4f} m')
print(f'final position error: Part I {e1[-1]:.3f} m, Part II {e2[-1]:.3f} m; '
      f'final heading error Part II {np.degrees(wrap(X[-1, 2] - gt[-1, 2])):.2f} deg')
print('max NIS', lm_nis.max(), 'count > 15:', int(np.sum(lm_nis > 15)))
