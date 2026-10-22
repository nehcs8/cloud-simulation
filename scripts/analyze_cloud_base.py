"""
Cloud-base organisation of single_cu runs (hot spot / heated hill at the domain centre).

Per snapshot (every EVERY-th): cloud base CB near the centre (r < 3.5 km), number of separate cloud
pieces 150 m and 550 m above CB and the largest one's equivalent diameter, updraft cores (w > 1 m/s)
800 m above the ground, low cloud (< 2.5 km) more than 4.5 km from the centre, and the
near-ground radial wind toward the centre (u_r < 0 = inflow) at 50 m above the ground in rings
r = 0.5–1.5, 1.5–3, 3–4.5 km (upslope flow / convergence over a heated hill).

Usage: python scripts/analyze_cloud_base.py RUN_DIR [RUN_DIR ...] [--every N]
(default N = 15 snapshots, i.e. 30 min at 120 s snapshots).
"""

import glob
import os
import sys

import numpy as np
from scipy.ndimage import label


def eq_diam(sizes, dx):
    return 2 * np.sqrt(sizes.max() * dx * dx / np.pi) if len(sizes) else 0.0


def effective(run, key, default):
    try:
        for line in open(os.path.join(run, 'effective_config.txt')):
            if line.startswith(key + ' ='):
                return eval(line.split('=', 1)[1].strip())
    except (OSError, SyntaxError, NameError):
        pass
    return default


def main():
    args = sys.argv[1:]
    every = 15
    if '--every' in args:
        i = args.index('--every'); every = int(args[i + 1]); del args[i:i + 2]
    for d in args:
        print('==', os.path.basename(d.rstrip('/')))
        dx = effective(d, 'DX', 100.0)
        h_max, sigma = effective(d, 'TERRAIN_H_MAX', 0.0), effective(d, 'TERRAIN_SIGMA', 1.0)
        hill = effective(d, 'TERRAIN_TYPE', 'flat') == 'gaussian_hill'
        for p in sorted(glob.glob(os.path.join(d, 'snapshots', 'snap_*.npz')))[every::every]:
            s = np.load(p)
            qc, w, u, v, z = s['qc'], s['w'], s['u'], s['v'], s['z']
            nx, ny = qc.shape[:2]
            X, Y = np.meshgrid((np.arange(nx) + .5) * dx - nx * dx / 2,
                               (np.arange(ny) + .5) * dx - ny * dx / 2, indexing='ij')
            rr = np.hypot(X, Y)
            near = rr < 3500
            h = h_max * np.exp(-rr**2 / (2 * sigma**2)) if hill else np.zeros_like(rr)

            # radial wind 50 m above the ground (nearest level per column)
            k50 = np.clip(np.searchsorted(z, h + 50.0), 0, len(z) - 1)
            ur = (u[np.arange(nx)[:, None], np.arange(ny)[None, :], k50] * X
                  + v[np.arange(nx)[:, None], np.arange(ny)[None, :], k50] * Y) / np.maximum(rr, 1.0)
            rings = ' '.join(f'{ur[(rr >= a) & (rr < b)].mean():+.2f}'
                             for a, b in ((500, 1500), (1500, 3000), (3000, 4500)))

            k8 = np.clip(np.searchsorted(z, h + 800.0), 0, len(z) - 1)      # 800 m above the ground
            w8 = w[np.arange(nx)[:, None], np.arange(ny)[None, :], k8]
            lab, n_core = label((w8 > 1.0) & near)
            core = f"cores@800m AGL {n_core:2d}, largest {eq_diam(np.bincount(lab.ravel())[1:], dx):4.0f} m"

            cl = qc > 1e-5
            t = int(s['t']) // 60
            if not cl[near].any():
                print(f"  t={t:3d} clear near centre | {core} | u_r@50m AGL {rings}")
                continue
            cb = z[np.nonzero(cl[near].any(axis=0))[0][0]]
            out = []
            for dzb in (150, 550):
                k = np.searchsorted(z, cb + dzb)
                lab, n = label(cl[:, :, k] & near)
                out.append(f"base+{dzb}: {n:2d} pieces, largest {eq_diam(np.bincount(lab.ravel())[1:], dx):4.0f} m")
            lowfar = (cl[:, :, :np.searchsorted(z, 2500)].any(axis=2) & (rr > 4500)).sum()
            print(f"  t={t:3d} CB {cb:4.0f} | " + ' | '.join(out) + f" | {core} | far low-cloud cols {lowfar:3d}"
                  f" | u_r@50m AGL {rings}")


if __name__ == "__main__":
    main()
