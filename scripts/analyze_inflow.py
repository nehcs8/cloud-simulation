"""
Inflow quality of runs with a mean wind along +x: how the state changes along the fetch.

Per x-bin (--bin m, default 800 m), averaged over y and the snapshots in [T0, T1] (min):
mixed-layer θ′ and q_v (mean below --zml m), resolved w′² at --zw m (deviation from the domain-level mean),
cloud frequency (column q_c > 1e-5) and its mean base. For an ideal inflow every column is the same along x.
Summary per run: θ′/q_v difference last − first bin, the distance where w′² first reaches 80 % of its
downstream-half mean ("spin-up fetch"), and the cloud fraction in the upstream vs downstream half.
With OPEN_INFLOW_REF = "mean" the snapshots carry open_ref_theta/open_ref_qv: printed against the domain mean.

Run: ./venv/bin/python3 scripts/analyze_inflow.py RUN_DIR [RUN_DIR ...] [--t0 60] [--t1 180] [--bin 800]
"""
import argparse
import glob
import os
import sys

import warnings

import numpy as np

warnings.filterwarnings("ignore", "Mean of empty slice")


def load(run, t0, t1):
    files = sorted(glob.glob(os.path.join(run, 'snapshots', 'snap_*.npz')))
    out = []
    for f in files:
        t = float(os.path.basename(f)[5:-4]) / 60
        if t0 <= t <= t1 and round(t) % 15 == 0:      # 15-min sampling: same for thinned and 1-min runs
            out.append(np.load(f))
    return out


def along_x(snaps, nb, zml, zw):
    z = snaps[0]['z'].astype(float)
    kml, kw = z < zml, np.argmin(abs(z - zw))
    acc = {k: [] for k in ('th', 'qv', 'w2', 'cf', 'cb')}
    for d in snaps:
        nx = d['theta'].shape[0]
        b = lambda a: a.reshape(nb, nx // nb, *a.shape[1:]).mean(axis=(1, 2))
        acc['th'].append(b(d['theta'][:, :, kml].mean(2)))
        acc['qv'].append(b(d['qv'][:, :, kml].mean(2)) * 1e3)
        w = d['w'][:, :, kw]
        acc['w2'].append(b((w - w.mean()) ** 2))
        cl = d['qc'] > 1e-5
        col = cl.any(2)
        acc['cf'].append(b(col.astype(float)) * 100)
        base = np.where(col, z[np.argmax(cl, axis=2)], np.nan)
        with np.errstate(invalid='ignore'):
            acc['cb'].append(np.nanmean(base.reshape(nb, -1), axis=1) if col.any() else np.full(nb, np.nan))
    return {k: (np.nanmean(v, axis=0) if k == 'cb' else np.mean(v, axis=0)) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--t0', type=float, default=60.0)
    ap.add_argument('--t1', type=float, default=180.0)
    ap.add_argument('--bin', type=float, default=800.0)
    ap.add_argument('--zml', type=float, default=600.0)
    ap.add_argument('--zw', type=float, default=600.0)
    a = ap.parse_args()
    for run in a.runs:
        snaps = load(run, a.t0, a.t1)
        if not snaps:
            print(f"{run}: no snapshots in window"); continue
        dx = 100.0
        cfgf = os.path.join(run, 'effective_config.txt')
        for line in open(cfgf):
            if line.startswith('DX ='):
                dx = float(line.split('=')[1])
        nx = snaps[0]['theta'].shape[0]
        nb = int(round(nx * dx / a.bin))
        r = along_x(snaps, nb, a.zml, a.zw)
        xc = (np.arange(nb) + 0.5) * nx * dx / nb / 1000
        print(f"\n{os.path.basename(run)}  ({len(snaps)} snapshots, {a.t0:.0f}–{a.t1:.0f} min)")
        print("  x [km] | θ′ ML [K] | q_v ML [g/kg] | w′² {:.0f} m | cloud % | base [m]".format(a.zw))
        for i in range(nb):
            print(f"  {xc[i]:6.1f} | {r['th'][i]:+8.3f}  | {r['qv'][i]:8.3f}     | {r['w2'][i]:6.3f}   |"
                  f" {r['cf'][i]:6.1f}  | {r['cb'][i]:6.0f}")
        half = nb // 2
        ref = r['w2'][half:].mean()
        reach = np.argmax(r['w2'] >= 0.8 * ref) if (r['w2'] >= 0.8 * ref).any() else -1
        print(f"  Δθ′ ML last−first bin {r['th'][-1] - r['th'][0]:+.3f} K, Δq_v {r['qv'][-1] - r['qv'][0]:+.3f} g/kg; "
              f"w′² reaches 80 % of the downstream mean ({ref:.3f}) at x ≈ {xc[reach]:.1f} km; "
              f"cloud upstream/downstream half {r['cf'][:half].mean():.1f} / {r['cf'][half:].mean():.1f} %")
        d = snaps[-1]
        if 'open_ref_theta' in d.files:
            z = d['z']; kml = z < a.zml
            print(f"  t = {float(d['t'])/60:.0f} min: inflow reference θ′ ML {d['open_ref_theta'][kml].mean():+.3f} K, "
                  f"q_v {d['open_ref_qv'][kml].mean()*1e3:.3f} g/kg vs domain mean {d['theta'][:, :, kml].mean():+.3f} K, "
                  f"{d['qv'][:, :, kml].mean()*1e3:.3f} g/kg")


if __name__ == '__main__':
    sys.exit(main())
