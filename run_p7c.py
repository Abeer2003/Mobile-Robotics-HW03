import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Ellipse
from scipy.stats import chi2

from ekf import wrap
from ekf_slam import EkfSlam

PAIR = 'P06'
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', '..', 'data', PAIR)
OUT = os.path.join(HERE, '..', 'plots')
STOP_EPS = 1e-9
GATE = chi2.ppf(0.99, 2)

run = pd.read_csv(os.path.join(DATA, 'run.csv'))
det = pd.read_csv(os.path.join(DATA, 'detections.csv'))
gp = yaml.safe_load(open(os.path.join(DATA, 'gyro_params.yaml')))
info = yaml.safe_load(open(os.path.join(DATA, 'run_info.yaml')))
lm = yaml.safe_load(open(os.path.join(HERE, '..', '..', 'config', 'landmarks.yaml')))['landmarks']
TRUE = {d['id']: np.array([d['x'], d['y']]) for d in lm}
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

ekf = EkfSlam(x0, P0, PARAMS)
X = np.zeros((n, 5))
Pd = np.zeros((n, 5))
n_lm = np.zeros(n, dtype=int)
lm_t, lm_nis, lm_idx, lm_true = [], [], [], []
created = []
kinds = []
snap = {}
for k in range(n):
    ekf.predict(ds[k], dth[k], om[k])
    if still[k]:
        ekf.update_zero_rate(om[k])
    else:
        ekf.update_gyro(om[k], ds[k], dth[k])
    used = set()
    for i in by_step.get(k, []):
        d = det.iloc[i]
        kind, j, nis = ekf.process_detection(d.range, d.bearing, t[k], used, GATE)
        kinds.append(kind)
        if kind == 'update':
            lm_t.append(t[k])
            lm_nis.append(nis)
            lm_idx.append(j)
            lm_true.append(int(d.id))
        elif kind == 'new':
            created.append((t[k], j, int(d.id)))
            lm_idx.append(j)
            lm_true.append(int(d.id))
            lm_t.append(np.nan)
            lm_nis.append(np.nan)
    X[k] = ekf.x[:5]
    Pd[k] = np.diag(ekf.P[:5, :5])
    n_lm[k] = ekf.n
    if k % 500 == 0:
        snap[k] = (ekf.x[:2].copy(), ekf.P[:2, :2].copy())

lm_t, lm_nis = np.array(lm_t), np.array(lm_nis)
lm_idx, lm_true = np.array(lm_idx), np.array(lm_true)
ok = ~np.isnan(lm_nis)
est_xy = np.array([ekf.lm(j) for j in range(ekf.n)])
est_P = [ekf.P[5 + 2 * j:7 + 2 * j, 5 + 2 * j:7 + 2 * j] for j in range(ekf.n)]

vote = {}
for j in range(ekf.n):
    c = Counter(lm_true[lm_idx == j])
    vote[j] = c.most_common(1)[0][0]
wrong = int(np.sum([lm_true[a] != vote[lm_idx[a]] for a in range(len(lm_idx)) if ok[a]]))
true_seen = sorted(set(det.id.values))
mapped = Counter(vote.values())
dup = {k: v for k, v in mapped.items() if v > 1}
lm_err = np.array([np.linalg.norm(est_xy[j] - TRUE[vote[j]]) for j in range(ekf.n)])

r4 = np.load(os.path.join(OUT, 'p4_partII_result.npz'))
X4 = r4['X']
e4 = np.hypot(X4[:, 0] - gt[:, 0], X4[:, 1] - gt[:, 1])
e7 = np.hypot(X[:, 0] - gt[:, 0], X[:, 1] - gt[:, 1])
h_err = np.degrees(wrap(X[:, 2] - gt[:, 2]))

np.savez(os.path.join(OUT, 'p7c_result.npz'), t=t, X=X, Pd=Pd, n_lm=n_lm, est_xy=est_xy,
         lm_t=lm_t, lm_nis=lm_nis, vote=np.array([vote[j] for j in range(ekf.n)]))

nis = lm_nis[ok]
N = len(nis)
lo1, hi1 = chi2.ppf([0.025, 0.975], 2)
W = 20
ma = np.convolve(nis, np.ones(W) / W, mode='valid')
loW, hiW = chi2.ppf([0.025, 0.975], 2 * W) / W
loN, hiN = chi2.ppf([0.025, 0.975], 2 * N) / N
print('gate %.2f, second gate %.1f, candidate radius %.2f m, %d sightings to confirm' % (GATE, ekf.gate_new, ekf.cand_radius, ekf.cand_count))
print('detections: %s' % dict(Counter(kinds)))
print('landmarks created: %d, true landmarks seen: %d (ids %s)' % (ekf.n, len(true_seen), true_seen))
print('estimated landmark -> true id:', vote)
print('duplicates (true id with more than one estimated landmark):', dup)
print('wrong associations among %d updates: %d' % (N, wrong))
print('landmark position error: mean %.3f m, max %.3f m' % (lm_err.mean(), lm_err.max()))
print('landmark NIS: N=%d mean %.3f, 95%% bounds [%.3f, %.3f]' % (N, nis.mean(), loN, hiN))
print('per-step inside [%.3f, %.3f]: %.1f%%, MA inside [%.3f, %.3f]: %.1f%%' %
      (lo1, hi1, 100 * np.mean((nis >= lo1) & (nis <= hi1)), loW, hiW, 100 * np.mean((ma >= loW) & (ma <= hiW))))
print('position RMSE: SLAM %.4f m, Problem 4 (known map) %.4f m' % (np.sqrt(np.mean(e7 ** 2)), np.sqrt(np.mean(e4 ** 2))))
print('final position error: SLAM %.4f m, P4 %.4f m; max SLAM error %.3f m at t=%.1f s' %
      (e7[-1], e4[-1], e7.max(), t[e7.argmax()]))
print('final heading error SLAM %.2f deg' % h_err[-1])
print('landmarks created at times:', [(round(a, 1), b) for a, b, _ in created])


def ell(ax, mu, C, color, lw=0.9, scale=1.0):
    w, v = np.linalg.eigh(C)
    ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    ax.add_patch(Ellipse(mu, scale * 4 * np.sqrt(max(w[1], 0)), scale * 4 * np.sqrt(max(w[0], 0)),
                         angle=ang, fill=False, color=color, lw=lw))


fig, ax = plt.subplots(figsize=(9, 8))
ax.plot(gt[:, 0], gt[:, 1], 'g-', lw=3, label='ground truth path')
ax.plot(X[:, 0], X[:, 1], 'r--', lw=1, label='EKF-SLAM path')
for k, (mu, C) in snap.items():
    ell(ax, mu, C, 'b', 1.0, 10.0)
for j in range(ekf.n):
    ell(ax, est_xy[j], est_P[j], 'm', 1.2, 10.0)
ax.plot(est_xy[:, 0], est_xy[:, 1], 'm+', ms=7, label='estimated landmarks (2$\\sigma$)')
tr = np.array(list(TRUE.values()))
ax.plot(tr[:, 0], tr[:, 1], 'k^', ms=6, mfc='none', label='true landmarks')
ax.set_aspect('equal')
ax.set_title('EKF-SLAM: path and map (2$\\sigma$ ellipses enlarged x10, robot every 5 s)')
ax.grid(True)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.legend(loc='upper right')
fig.savefig(os.path.join(OUT, 'p7c_path_map.png'), dpi=150, bbox_inches='tight')

fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
axs[0].semilogy(lm_t[ok], np.maximum(nis, 1e-6), '.', ms=2)
axs[0].axhline(lo1, color='r', ls='--')
axs[0].axhline(hi1, color='r', ls='--')
axs[0].set_ylabel('NIS (2 DOF)')
axs[0].set_title('EKF-SLAM landmark updates: per-step NIS, chi2_2 95% bounds')
axs[1].plot(lm_t[ok][W - 1:], ma)
axs[1].axhline(loW, color='r', ls='--')
axs[1].axhline(hiW, color='r', ls='--')
axs[1].axhline(2.0, color='k', ls=':')
axs[1].set_ylabel('20-update moving average')
axs[1].set_xlabel('time (s)')
for a in axs:
    a.grid(True)
fig.savefig(os.path.join(OUT, 'p7c_nis.png'), dpi=150, bbox_inches='tight')

fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
axs[0].semilogy(t, np.maximum(e4, 1e-5), 'b', label='Problem 4 (known map)')
axs[0].semilogy(t, np.maximum(e7, 1e-5), 'r', label='EKF-SLAM (unknown map)')
axs[0].plot(t, 2 * np.sqrt(Pd[:, 0] + Pd[:, 1]), 'k--', lw=0.8, label='SLAM 2$\\sigma$ position')
axs[0].set_ylabel('position error (m)')
axs[0].legend()
axs[1].plot(t, n_lm, 'k')
axs[1].set_ylabel('landmarks in the map')
axs[1].set_xlabel('time (s)')
for a in axs:
    a.grid(True)
fig.savefig(os.path.join(OUT, 'p7c_error.png'), dpi=150, bbox_inches='tight')
