import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2

from ekf import Ekf, wrap

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


def run_filter(gate):
    ekf = Ekf(x0, P0, PARAMS)
    X = np.zeros((n, 5))
    P = np.zeros((n, 5, 5))
    lm_t, lm_nis, assigned, true = [], [], [], []
    for k in range(n):
        ekf.predict(ds[k], dth[k], om[k])
        if still[k]:
            ekf.update_zero_rate(om[k])
        else:
            ekf.update_gyro(om[k], ds[k], dth[k])
        for i in by_step.get(k, []):
            d = det.iloc[i]
            j, _ = ekf.associate(d.range, d.bearing, MAP, gate)
            assigned.append(-1 if j is None else j)
            true.append(int(d.id))
            if j is not None:
                lm_nis.append(ekf.update_landmark(MAP[j], d.range, d.bearing))
                lm_t.append(t[k])
        X[k], P[k] = ekf.x, ekf.P
    return X, P, np.array(lm_t), np.array(lm_nis), np.array(assigned), np.array(true)


def summary(X, assigned, true):
    correct = int(np.sum(assigned == true))
    missed = int(np.sum(assigned == -1))
    wrong = int(np.sum((assigned != -1) & (assigned != true)))
    e = np.hypot(X[:, 0] - gt[:, 0], X[:, 1] - gt[:, 1])
    return correct, missed, wrong, float(np.sqrt(np.mean(e ** 2))), e


print('gate sweep (detections %d)' % len(det))
for g in (chi2.ppf(0.95, 2), chi2.ppf(0.99, 2), chi2.ppf(0.999, 2), 30.0):
    Xg, _, _, nisg, ag, tg = run_filter(g)
    c, m, w, rmse, _ = summary(Xg, ag, tg)
    print('  gate %.2f: correct %d, missed %d, wrong %d, RMSE %.4f m, mean NIS %.3f' % (g, c, m, w, rmse, nisg.mean()))

X, P, lm_t, lm_nis, assigned, true = run_filter(GATE)
correct, missed, wrong, rmse6, e6 = summary(X, assigned, true)
np.savez(os.path.join(OUT, 'p6a_result.npz'), t=t, X=X, P=P, lm_t=lm_t, lm_nis=lm_nis,
         assigned=assigned, true=true)

r4 = np.load(os.path.join(OUT, 'p4_partII_result.npz'))
X4, nis4 = r4['X'], r4['lm_nis']
e4 = np.hypot(X4[:, 0] - gt[:, 0], X4[:, 1] - gt[:, 1])

print('gate = %.2f (chi2_2 99%%): correct %d, missed %d, wrong %d of %d' % (GATE, correct, missed, wrong, len(det)))
print('position RMSE: Problem 4 (known ids) %.4f m, Problem 6 %.4f m' % (np.sqrt(np.mean(e4 ** 2)), rmse6))
print('final position error: P4 %.4f m, P6 %.4f m' % (e4[-1], e6[-1]))

N = len(lm_nis)
lo1, hi1 = chi2.ppf([0.025, 0.975], 2)
W = 20
ma = np.convolve(lm_nis, np.ones(W) / W, mode='valid')
loW, hiW = chi2.ppf([0.025, 0.975], 2 * W) / W
loN, hiN = chi2.ppf([0.025, 0.975], 2 * N) / N
print('landmark NIS: P6 N=%d mean %.3f bounds [%.3f, %.3f]; P4 N=%d mean %.3f' %
      (N, lm_nis.mean(), loN, hiN, len(nis4), nis4.mean()))
print('per-step inside [%.3f, %.3f]: %.1f%%, MA inside [%.3f, %.3f]: %.1f%%' %
      (lo1, hi1, 100 * np.mean((lm_nis >= lo1) & (lm_nis <= hi1)), loW, hiW,
       100 * np.mean((ma >= loW) & (ma <= hiW))))
bad = np.flatnonzero((assigned != -1) & (assigned != true))
print('wrong association times (s):', np.round(det.time.values[bad][:20], 1))
miss = np.flatnonzero(assigned == -1)
print('missed association times (s):', np.round(det.time.values[miss][:20], 1))

fig, axs = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
for ax, Xe, title in [(axs[0], X4, 'Problem 4 (ids given)'), (axs[1], X, 'Problem 6 (nearest neighbour + gate)')]:
    ax.plot(gt[:, 0], gt[:, 1], 'g-', lw=3, label='ground truth')
    ax.plot(Xe[:, 0], Xe[:, 1], 'r--', lw=1, label='EKF')
    for m in MAP.values():
        ax.plot(m[0], m[1], 'k^', ms=5)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel('x (m)')
axs[0].set_ylabel('y (m)')
axs[0].legend(loc='upper right')
fig.savefig(os.path.join(OUT, 'p6a_path.png'), dpi=150, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(9, 4))
ax.semilogy(t, np.maximum(e4, 1e-5), 'b', label='Problem 4 (ids given)')
ax.semilogy(t, np.maximum(e6, 1e-5), 'r', label='Problem 6 (gated NN)')
ax.set_xlabel('time (s)')
ax.set_ylabel('position error (m)')
ax.legend()
ax.grid(True)
fig.savefig(os.path.join(OUT, 'p6a_error.png'), dpi=150, bbox_inches='tight')

fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
axs[0].semilogy(lm_t, np.maximum(lm_nis, 1e-6), '.', ms=2)
axs[0].axhline(lo1, color='r', ls='--')
axs[0].axhline(hi1, color='r', ls='--')
axs[0].set_ylabel('NIS (2 DOF)')
axs[0].set_title('Problem 6 landmark updates: per-step NIS, chi2_2 95% bounds')
axs[1].plot(lm_t[W - 1:], ma)
axs[1].axhline(loW, color='r', ls='--')
axs[1].axhline(hiW, color='r', ls='--')
axs[1].axhline(2.0, color='k', ls=':')
axs[1].set_ylabel('20-update moving average')
axs[1].set_xlabel('time (s)')
for a in axs:
    a.grid(True)
fig.savefig(os.path.join(OUT, 'p6a_nis.png'), dpi=150, bbox_inches='tight')
