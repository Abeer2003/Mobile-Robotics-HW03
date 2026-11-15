import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'plots')

r = np.load(os.path.join(OUT, 'p7c_result.npz'))
t, Pd = r['t'], r['Pd'][:, :3]
P0 = np.array([1e-6, 1e-6, 1e-6])
names = ['x', 'y', '$\\theta$']
units = ['m$^2$', 'm$^2$', 'rad$^2$']

fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
for i, a in enumerate(axs):
    a.semilogy(t, Pd[:, i], 'b', label='EKF-SLAM variance')
    a.axhline(P0[i], color='r', ls='--', label='initial value $P_0$')
    a.axvline(248, color='gray', ls=':', label='back at start pose (248 s)')
    a.set_ylabel('var(%s) (%s)' % (names[i], units[i]))
    a.grid(True)
axs[0].legend(ncol=3, fontsize=8)
axs[-1].set_xlabel('time (s)')
fig.savefig(os.path.join(OUT, 'p7d_variances.png'), dpi=150, bbox_inches='tight')

for i, nm in enumerate(['x', 'y', 'theta']):
    k = Pd[:, i].argmin()
    below = int(np.sum(Pd[:, i] < P0[i]))
    print('%-5s initial %.3e  minimum %.9e at t=%.2f s (min/initial %.6f)  steps below initial: %d  final %.3e  maximum %.3e' %
          (nm, P0[i], Pd[k, i], t[k], Pd[k, i] / P0[i], below, Pd[-1, i], Pd[:, i].max()))
