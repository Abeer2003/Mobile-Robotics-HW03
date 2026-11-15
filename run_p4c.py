
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
os.makedirs(OUT, exist_ok=True)

STOP_EPS = 1e-9   # still if |ds|, |dtheta| < this (the first 60 s are ~1e-12, not exactly 0)

run = pd.read_csv(os.path.join(DATA, 'run.csv'))
gp = yaml.safe_load(open(os.path.join(DATA, 'gyro_params.yaml')))
info = yaml.safe_load(open(os.path.join(DATA, 'run_info.yaml')))
g60 = pd.read_csv(os.path.join(DATA, 'gyro_60s.csv')).gyro_z.values
dt = gp['dt']
BASE = dict(sigma_n=gp['sigma_n'], sigma_g=gp['sigma_g'], tau=gp['tau'],
            k1=info['odometry_noise']['k1'], k2=info['odometry_noise']['k2'], dt=dt)

B0_HAT = float(g60.mean())
VAR_B0 = gp['sigma_g'] ** 2
POSE_STD = (1e-3, 1e-3, 1e-3)     # m, m, rad: initial pose "known"
x0 = [info['initial_pose']['x'], info['initial_pose']['y'], info['initial_pose']['theta'], B0_HAT, 0.0]
P0 = np.diag([POSE_STD[0] ** 2, POSE_STD[1] ** 2, POSE_STD[2] ** 2, VAR_B0, gp['sigma_g'] ** 2])

t = run.time.values
ds, dth, om = run.ds.values, run.dtheta.values, run.gyro_z.values
gt = run[['gt_x', 'gt_y', 'gt_theta']].values
n = len(run)
still = (np.abs(ds) < STOP_EPS) & (np.abs(dth) < STOP_EPS)


def run_filter(**over):
    ekf = Ekf(x0, P0, {**BASE, **over})
    X, P = np.zeros((n, 5)), np.zeros((n, 5, 5))
    nis = np.full(n, np.nan)
    for k in range(n):
        ekf.predict(ds[k], dth[k], om[k])
        if still[k]:
            nis[k] = ekf.update_zero_rate(om[k])
        else:
            ekf.update_gyro(om[k], ds[k], dth[k])
        X[k], P[k] = ekf.x, ekf.P
    return X, P, nis


# ---- main run -----------------------------------------------------------------------------------
X, P, nis = run_filter()
np.savez(os.path.join(OUT, 'p4_partI_result.npz'), t=t, X=X, P=P, nis=nis)

# path with 2-sigma ellipses every 5 s
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(gt[:, 0], gt[:, 1], 'g-', label='ground truth')
ax.plot(X[:, 0], X[:, 1], 'r-', label='EKF (wheels + gyro)')
for k in range(0, n, 500):
    w, v = np.linalg.eigh(P[k, :2, :2])
    ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    ax.add_patch(Ellipse(X[k, :2], 4 * np.sqrt(max(w[1], 0)), 4 * np.sqrt(max(w[0], 0)),
                         angle=ang, fill=False, color='b', lw=0.8))
ax.set_aspect('equal'); ax.grid(True); ax.legend()
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_title('Path with 2$\\sigma$ ellipses every 5 s')
fig.savefig(os.path.join(OUT, 'p4c_path.png'), dpi=150, bbox_inches='tight')

# pose components vs ground truth with +/-2 sigma; heading error wrapped
fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
for i, (a, lab) in enumerate(zip(axs, ['x (m)', 'y (m)', '$\\theta$ (rad)'])):
    s = np.sqrt(P[:, i, i])
    est, tru = X[:, i], gt[:, i]
    if i == 2:   # unwrap both heading curves (same branch) so no +-pi jump appears
        est, tru = np.unwrap(est), np.unwrap(tru)
        est = est + 2 * np.pi * np.round((tru[0] - est[0]) / (2 * np.pi))
    a.plot(t, est, 'r', label='estimate')
    a.fill_between(t, est - 2 * s, est + 2 * s, color='r', alpha=0.25, label='$\\pm2\\sigma$')
    a.plot(t, tru, 'g', label='ground truth')
    a.set_ylabel(lab); a.grid(True)
axs[0].legend(ncol=3); axs[-1].set_xlabel('time (s)')
fig.savefig(os.path.join(OUT, 'p4c_pose.png'), dpi=150, bbox_inches='tight')

fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
for i, (a, lab) in enumerate(zip(axs, ['x error (m)', 'y error (m)', '$\\theta$ error (rad, wrapped)'])):
    e = (X[:, i] - gt[:, i]) if i < 2 else wrap(X[:, 2] - gt[:, 2])
    s = np.sqrt(P[:, i, i])
    a.plot(t, e, 'r'); a.fill_between(t, -2 * s, 2 * s, color='r', alpha=0.25)
    a.set_ylabel(lab); a.grid(True)
axs[-1].set_xlabel('time (s)')
fig.savefig(os.path.join(OUT, 'p4c_pose_error.png'), dpi=150, bbox_inches='tight')

# b0, g, b0+g with +/-2 sigma
tot = X[:, 3] + X[:, 4]
vtot = P[:, 3, 3] + P[:, 4, 4] + 2 * P[:, 3, 4]
fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
for a, y, v, lab in [(axs[0], X[:, 3], P[:, 3, 3], '$\\hat b_0$ (rad/s)'),
                     (axs[1], X[:, 4], P[:, 4, 4], '$\\hat g_k$ (rad/s)'),
                     (axs[2], tot, vtot, '$\\hat b_0+\\hat g_k$ (rad/s)')]:
    s = np.sqrt(v)
    a.plot(t, y, 'r'); a.fill_between(t, y - 2 * s, y + 2 * s, color='r', alpha=0.25)
    a.set_ylabel(lab); a.grid(True)
axs[-1].set_xlabel('time (s)')
fig.savefig(os.path.join(OUT, 'p4c_bias.png'), dpi=150, bbox_inches='tight')

def nis_report(nis, name, fname):
    v = nis[~np.isnan(nis)]
    tv = t[~np.isnan(nis)]
    N = len(v)
    lo1, hi1 = chi2.ppf([0.025, 0.975], 1)
    W = 20
    ma = np.convolve(v, np.ones(W) / W, mode='valid')
    loW, hiW = chi2.ppf([0.025, 0.975], W) / W
    loN, hiN = chi2.ppf([0.025, 0.975], N) / N
    print(f'[{name}] N={N} updates, time-averaged NIS = {v.mean():.3f}, '
          f'95% bounds for the mean [{loN:.3f}, {hiN:.3f}], '
          f'inside per-step bounds: {np.mean((v >= lo1) & (v <= hi1)) * 100:.1f}%, '
          f'inside MA bounds: {np.mean((ma >= loW) & (ma <= hiW)) * 100:.1f}%')
    fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axs[0].semilogy(tv, np.maximum(v, 1e-6), '.', ms=1.5)
    axs[0].axhline(lo1, color='r', ls='--'); axs[0].axhline(hi1, color='r', ls='--')
    axs[0].set_ylabel('NIS (1 DOF)'); axs[0].set_title(f'{name}: per-step NIS, $\\chi^2_1$ 95% bounds')
    axs[1].plot(tv[W - 1:], ma)
    axs[1].axhline(loW, color='r', ls='--'); axs[1].axhline(hiW, color='r', ls='--')
    axs[1].axhline(1.0, color='k', ls=':')
    axs[1].set_ylabel('20-update moving average'); axs[1].set_xlabel('time (s)')
    for a in axs: a.grid(True)
    fig.savefig(os.path.join(OUT, fname), dpi=150, bbox_inches='tight')


nis_report(nis, 'baseline', 'p4d_nis.png')
_, _, nis_q = run_filter(sigma_g_q=gp['sigma_g'] / 10)
nis_report(nis_q, 'sigma_g/10 (Q too small)', 'p4e_nis_Qsmall.png')
_, _, nis_r = run_filter(R_scale=0.01)      # R 100x too small = sigma_n/10
nis_report(nis_r, 'R too small (sigma_n/10)', 'p4e_nis_Rsmall.png')

e = np.hypot(X[:, 0] - gt[:, 0], X[:, 1] - gt[:, 1])
print(f'final position error {e[-1]:.3f} m, RMSE {np.sqrt(np.mean(e**2)):.3f} m, '
      f'final heading error {np.degrees(wrap(X[-1,2]-gt[-1,2])):.2f} deg')
print(f'b0_hat(60 s) = {X[5999,3]:.5f}, std {np.sqrt(P[5999,3,3]):.5f}; total bias at end {tot[-1]:.5f}')
print('figures in', OUT)
