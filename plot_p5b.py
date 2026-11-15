import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else '.'
X0, Y0 = -3.5, -3.0
load = lambda f: np.genfromtxt(os.path.join(D, f), delimiter=',', names=True)
ekf, gt, own = load('ekf.csv'), load('gt.csv'), load('own.csv')


def err(t, x, y):
    gx = np.interp(t, gt['time'], gt['x'])
    gy = np.interp(t, gt['time'], gt['y'])
    return np.hypot(x - gx, y - gy)


t0 = ekf['time'][0]
e_f = err(ekf['time'], ekf['x'], ekf['y'])
e_o = err(own['time'], own['x'] + X0, own['y'] + Y0)
sig = np.sqrt(ekf['Pxx'] + ekf['Pyy'])
rms_f = np.sqrt(np.mean(e_f ** 2))
rms_o = np.sqrt(np.mean(e_o ** 2))
print('RMS position error  filter %.3f m   /odom_own %.3f m' % (rms_f, rms_o))

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(own['time'] - t0, e_o, color='tab:blue', label='/odom_own (stage 1), RMS %.3f m' % rms_o)
ax.plot(ekf['time'] - t0, e_f, color='tab:red', label='/ekf/odom (stage 2), RMS %.3f m' % rms_f)
ax.plot(ekf['time'] - t0, 2 * sig, 'k--', label='filter 2 sigma position band')
ax.set_xlabel('time (s)')
ax.set_ylabel('position error (m)')
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(D, 'p5b_position_error.png'), dpi=150)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(gt['x'], gt['y'], color='tab:green', lw=3, label='ground truth')
ax.plot(own['x'] + X0, own['y'] + Y0, color='tab:blue', label='/odom_own')
ax.plot(ekf['x'], ekf['y'], color='tab:red', ls='--', label='/ekf/odom')
ax.set_aspect('equal')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(D, 'p5b_path.png'), dpi=150)
