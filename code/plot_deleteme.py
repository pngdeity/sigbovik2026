#!/usr/bin/env python3
"""Plot results from archived npz files with scienceplots styling."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

TIMESTAMPS = [
    '2026-03-19T20:21:48.423721+00:00',
    '2026-03-19T21:14:12.377259+00:00',
    '2026-03-19T11:45:15.633743+00:00',
]

for ts in TIMESTAMPS:
    path = f'/www/flatearth/archive_vars/{ts}.npz'
    data = np.load(path)

    b_np    = data['b_opt']
    r_src   = data['r_src']
    gz_np   = data['gz']
    gr_np   = data['gr']
    err_np  = data['err']
    r_obs   = data['r_obs']
    epsilon = float(data['epsilon'])
    g0      = float(data['g0'])
    disk_r  = float(data['disk_r'])
    n_src   = int(data['n_src'])
    n_obs   = int(data['n_obs'])
    n_z     = int(data['n_z'])
    smoothing = float(data['smoothing'])
    R_ext   = float(data['R_ext'])

    dr   = float(R_ext / n_src)
    mass = float(2 * np.pi * (b_np * r_src * dr).sum())

    # For the run that achieved ε_max=0.00094, display ε=0.001
    display_epsilon = 0.001 if (err_np.max() < 0.001 and epsilon < 0.001) else epsilon

    fig, axes = plt.subplots(2, 1, figsize=(4, 4), gridspec_kw={'height_ratios': [1, 2]})

    # Panel 1: slab cross-section
    ax = axes[0]
    r_plot = np.concatenate([-r_src[::-1], r_src])
    b_plot = np.concatenate([b_np[::-1], b_np])
    ax.fill_between(r_plot, 0, -b_plot, color='steelblue', alpha=0.8)
    ax.plot(r_plot, -b_plot, color='black', linewidth=0.1)
    ax.plot([-disk_r, disk_r], [0, 0], 'r-', lw=3, label='Disk')
    ax.axvline( disk_r, color='r', ls='--', alpha=0.3)
    ax.axvline(-disk_r, color='r', ls='--', alpha=0.3)
    ax.set_xlim(-R_ext / 2, R_ext / 2)
    ax.set_ylim(-b_np.max() * 1.3, b_np.max() * 0.3)

    ax.set_title(fr'Min-mass slab ($\varepsilon={display_epsilon}$)')
    ax.set_xlabel(r'$r$')
    ax.set_ylabel(r'$z$')
    ax.legend()

    # Panel 2: error
    ax = axes[1]
    ax.plot(r_obs, err_np, 'k-')
    ax.axhline(display_epsilon, color='r', ls='--',
               label=fr'$\varepsilon={display_epsilon}$')
    ax.set_ylim([0, 1.1 * display_epsilon])
    ax.set_title(r'Gravity Deviation $||g(r) - g_0||/||g_0||$')
    ax.set_xlabel(r'$r$')
    ax.legend()

    label = f'n_src={n_src}  n_obs={n_obs}  n_z={n_z}  smooth={smoothing:.2e}  mass={mass:.4f}'
    fig.text(0.01, 0.01, label, fontsize=8, color='gray', va='bottom', ha='left')
    plt.tight_layout()

    outpath = f'/www/flatearth/figure_{display_epsilon}.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Saved {outpath}")
    plt.close(fig)
