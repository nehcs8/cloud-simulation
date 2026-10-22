"""
Updraft dilution and local rain of single_cu runs.

Undiluted fraction of updraft cores (w > W_CORE and cloudy) at 2, 3, 4 km, from total water used as a
mixing tracer between mixed-layer air and the environment at that height:
    f = (q_core − q_env(z)) / (q_ml − q_env(z))
q = q_v + q_c (rain left out: it falls through the core) and q = q_v + q_c + q_r (upper bound).
q_env(z) = mean of the boundary columns (inflow air = sounding), q_ml = the same at the lowest level.
f ≈ 1: undiluted surface air (parcel theory); real congestus cores at mid-levels are about 0.3–0.6.

Also: w_max, rain water max, and the local surface rain rate R = ρ q_r V_t at the lowest level
(Kessler fall speed), max over the snapshots.

Usage: python scripts/analyze_updraft_dilution.py RUN_DIR [RUN_DIR ...]
"""

import glob
import os
import sys

import numpy as np

W_CORE = 5.0                       # m/s
LEVELS = (2000.0, 3000.0, 4000.0)  # m


def main():
    for d in sys.argv[1:]:
        snaps = sorted(glob.glob(os.path.join(d, 'snapshots', 'snap_*.npz')))
        print('==', os.path.basename(d.rstrip('/')))
        rain_max, rain_t = 0.0, 0
        for p in snaps:
            s = np.load(p)
            qv, qc, qr, w, z = s['qv'], s['qc'], s['qr'], s['w'], s['z']
            R = 1.15 * qr[:, :, 0] * 36.34 * (1.15 * qr[:, :, 0] / 1000.0) ** 0.1364 * 3600.0   # mm/h
            if R.max() > rain_max:
                rain_max, rain_t = float(R.max()), int(s['t']) // 60
            edge = np.zeros(qv.shape[:2], bool)
            edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
            parts = []
            for zz in LEVELS:
                k = int(np.searchsorted(z, zz))
                core = (w[:, :, k] > W_CORE) & (qc[:, :, k] > 1e-5)
                if core.sum() < 3:
                    parts.append(f"{zz/1e3:.0f} km: –")
                    continue
                f = []
                for q in (qv + qc, qv + qc + qr):
                    env, ml = q[:, :, k][edge].mean(), q[:, :, 0][edge].mean()
                    f.append(((q[:, :, k][core] - env) / (ml - env)).mean())
                parts.append(f"{zz/1e3:.0f} km: f {f[0]:.2f} ({f[1]:.2f}) n={int(core.sum())}")
            print(f"  t={int(s['t'])//60:3d}  w_max {w.max():5.1f}  qr_max {qr.max()*1e3:5.2f} | " + ' | '.join(parts))
        print(f"  local surface rain max {rain_max:.1f} mm/h at {rain_t} min")


if __name__ == "__main__":
    main()
