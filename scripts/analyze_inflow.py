"""
Inflow quality of open-boundary runs: how the state changes along the fetch.

The fetch of a column is its distance from the upwind open boundaries along the mean wind at --zw (from the
snapshots): x for a wind along +x, min(x / cos φ, y / sin φ) for a diagonal wind (since 2026-10-28; before, x only).
Runs without an open direction (periodic controls) are binned by x.
Per fetch bin (--bin m, default 800 m), averaged over the columns in the bin and the snapshots in [T0, T1] (min):
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


def fetch(snaps, dx, dy, bc_x, bc_y, zw):
    """(nx, ny) distance [m] of each column from the upwind open boundaries along the mean wind at zw
    (x if no direction is open), and a label for the printout."""
    d = snaps[0]
    nx, ny = d['theta'].shape[:2]
    x = (np.arange(nx) + 0.5)[:, None] * dx * np.ones((1, ny))
    y = (np.arange(ny) + 0.5)[None, :] * dy * np.ones((nx, 1))
    if bc_x != 'open' and bc_y != 'open':
        return x, 'x [km]'
    kw = np.argmin(abs(d['z'].astype(float) - zw))
    u = np.mean([s['u'][:, :, kw].mean() for s in snaps])
    v = np.mean([s['v'][:, :, kw].mean() for s in snaps])
    sp = np.hypot(u, v)
    F = np.full((nx, ny), np.inf)
    if bc_x == 'open' and abs(u) > 0.1 * sp:
        F = np.minimum(F, (x if u > 0 else nx * dx - x) * sp / abs(u))
    if bc_y == 'open' and abs(v) > 0.1 * sp:
        F = np.minimum(F, (y if v > 0 else ny * dy - y) * sp / abs(v))
    return F, f'fetch [km] (wind {u:+.1f}, {v:+.1f} m/s)'


def along_fetch(snaps, F, edges, zml, zw):
    z = snaps[0]['z'].astype(float)
    kml, kw = z < zml, np.argmin(abs(z - zw))
    idx = np.clip(np.digitize(F, edges) - 1, 0, len(edges) - 2).ravel()
    nb = len(edges) - 1
    n = np.bincount(idx, minlength=nb)
    b = lambda a: np.bincount(idx, a.ravel(), minlength=nb) / np.maximum(n, 1)
    acc = {k: [] for k in ('th', 'qv', 'w2', 'cf', 'cb')}
    for d in snaps:
        acc['th'].append(b(d['theta'][:, :, kml].mean(2)))
        acc['qv'].append(b(d['qv'][:, :, kml].mean(2)) * 1e3)
        w = d['w'][:, :, kw]
        acc['w2'].append(b((w - w.mean()) ** 2))
        cl = d['qc'] > 1e-5
        col = cl.any(2)
        acc['cf'].append(b(col.astype(float)) * 100)
        base = z[np.argmax(cl, axis=2)]
        nc = np.bincount(idx, col.ravel(), minlength=nb)
        acc['cb'].append(np.where(nc > 0, np.bincount(idx, (base * col).ravel(), minlength=nb) / np.maximum(nc, 1), np.nan))
    return {k: (np.nanmean(v, axis=0) if k == 'cb' else np.mean(v, axis=0)) for k, v in acc.items()}, n


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
        c = dict(DX=100.0, DY=100.0, BC_X='periodic', BC_Y='periodic')
        for line in open(os.path.join(run, 'effective_config.txt')):
            key, _, val = line.partition(' = ')
            if key in c:
                c[key] = type(c[key])(val.strip().strip("'"))
        F, label = fetch(snaps, c['DX'], c['DY'], c['BC_X'], c['BC_Y'], a.zw)
        fmax = F[np.isfinite(F)].max()
        nb = max(int(round(fmax / a.bin)), 1)
        edges = np.arange(nb + 1) * a.bin
        r, ncol = along_fetch(snaps, F, edges, a.zml, a.zw)
        keep = ncol > 0
        r = {k: v[keep] for k, v in r.items()}
        xc = ((edges[:-1] + 0.5 * a.bin) / 1000)[keep]
        nb = int(keep.sum())
        print(f"\n{os.path.basename(run)}  ({len(snaps)} snapshots, {a.t0:.0f}–{a.t1:.0f} min; bins by {label})")
        print("  {:>6} | θ′ ML [K] | q_v ML [g/kg] | w′² {:.0f} m | cloud % | base [m] | columns".format(
            label.split()[0], a.zw))
        for i in range(nb):
            print(f"  {xc[i]:6.1f} | {r['th'][i]:+8.3f}  | {r['qv'][i]:8.3f}     | {r['w2'][i]:6.3f}   |"
                  f" {r['cf'][i]:6.1f}  | {r['cb'][i]:6.0f}   | {ncol[keep][i]:5d}")
        half = nb // 2
        ref = r['w2'][half:].mean()
        reach = np.argmax(r['w2'] >= 0.8 * ref) if (r['w2'] >= 0.8 * ref).any() else -1
        print(f"  Δθ′ ML last−first bin {r['th'][-1] - r['th'][0]:+.3f} K, Δq_v {r['qv'][-1] - r['qv'][0]:+.3f} g/kg; "
              f"w′² reaches 80 % of the downstream mean ({ref:.3f}) at {label.split()[0]} ≈ {xc[reach]:.1f} km; "
              f"cloud upstream/downstream half {r['cf'][:half].mean():.1f} / {r['cf'][half:].mean():.1f} %")
        d = snaps[-1]
        if 'open_ref_theta' in d.files:
            z = d['z']; kml = z < a.zml
            print(f"  t = {float(d['t'])/60:.0f} min: inflow reference θ′ ML {d['open_ref_theta'][kml].mean():+.3f} K, "
                  f"q_v {d['open_ref_qv'][kml].mean()*1e3:.3f} g/kg vs domain mean {d['theta'][:, :, kml].mean():+.3f} K, "
                  f"{d['qv'][:, :, kml].mean()*1e3:.3f} g/kg")


if __name__ == '__main__':
    sys.exit(main())
