"""
Compare the microphysics of several runs (warm / ICE_SCHEME "mixed" / "graupel").

Per run: timing of rain aloft and at the ground, peaks from diag.csv, accumulated surface precipitation
(domain mean). Per snapshot (every --every minutes from --t0): hydrometeor masses (cloud liquid/ice
from the α(T) split of q_c, rain, graupel), graupel mean height, export of rain + graupel through the
lateral edges (cell-centred u, v of the edge cells — an estimate), cloud cover above 5 km, coldest
lowest-level θ′ (cold pool) and peak local surface rain (rain only, ρ q_r V_t at the lowest level).

With micro_budget.npz (runs from 2026-10-23 on), --budget T0 T1 [min] adds the process budget of that window:
mass converted by each process [kt], its mass-weighted mean height, and the fractions of the precipitation
formed (cloud → rain + cloud ice/liquid → graupel) lost to evaporation/sublimation and reaching the ground.

Run: ./venv/bin/python3 scripts/compare_microphysics.py RUN_DIR [RUN_DIR ...] [--t0 60] [--every 15] [--budget 0 150]
"""
import argparse
import glob
import os
import sys

import numpy as np

G, CP, RD = 9.81, 1004.0, 287.0


def reference(run, files):
    """θ0(z), π0(z), ρ0(z), dz(z), dx of a run (θ0 from profiles.npz minus the first snapshot's θ′)."""
    P = np.load(os.path.join(run, 'profiles.npz'))
    s0 = np.load(files[0])
    z = s0['z'].astype(float)
    th = P['theta'][0] if P['theta'].ndim == 2 else P['theta']
    th0 = th - s0['prof_theta']
    pi = np.ones_like(z)
    pi[0] = 1 - G * z[0] / (CP * th0[0])
    for k in range(1, len(z)):
        pi[k] = pi[k - 1] - G * (z[k] - z[k - 1]) / (CP * 0.5 * (th0[k] + th0[k - 1]))
    rho = 1e5 * pi ** (CP / RD - 1) / (RD * th0)
    dx = 100.0
    cfgf = os.path.join(run, 'effective_config.txt')
    if os.path.exists(cfgf):
        for line in open(cfgf):
            if line.startswith('DX ='):
                dx = float(line.split('=')[1])
    return z, th0, pi, rho, np.diff(s0['z_face']).astype(float), dx


def summary(run):
    d = np.genfromtxt(os.path.join(run, 'diag.csv'), delimiter=',', names=True)
    t = d['t'] / 60
    first = lambda m: f"{t[np.where(m)[0][0]]:.0f}" if m.any() else "-"
    qg = d['qg_max'].max() if 'qg_max' in d.dtype.names else 0.0
    print(f"{os.path.basename(run)}\n  rain aloft (q_r > 0.1 g/kg) {first(d['qr_max'] > 0.1)} min, at the ground "
          f"{first(d['precip_mmhr'] > 0.01)} min | w_max {d['w_max'].max():.1f} m/s, top {d['CT'].max():.0f} m, "
          f"q_r {d['qr_max'].max():.2f}, q_g {qg:.2f} g/kg | precip peak {d['precip_mmhr'].max():.2f} mm/h "
          f"(domain mean) at {t[d['precip_mmhr'].argmax()]:.0f} min, total {np.trapezoid(d['precip_mmhr'], t / 60):.3f} mm")


def edge_export(q, u, v, rho, dz, dx):
    """Outward flux of q through the four lateral edges [kg/s], from the edge cells' centred u, v."""
    A = (rho * dz)[None, :] * dx
    return float((np.maximum(-u[0], 0) * q[0] * A).sum() + (np.maximum(u[-1], 0) * q[-1] * A).sum()
                 + (np.maximum(-v[:, 0], 0) * q[:, 0] * A).sum() + (np.maximum(v[:, -1], 0) * q[:, -1] * A).sum())


def snapshots(run, t0, every):
    files = sorted(glob.glob(os.path.join(run, 'snapshots', 'snap_*.npz')))
    z, th0, pi, rho, dz, dx = reference(run, files)
    col = rho * dz * dx * dx / 1e6                     # kg/kg → kt per cell
    print("   t | cloud liq  ice   rain graupel [kt] | z_g [m] | edge export r+g [kt/5min] | cover>5km | θ′_sfc min | local rain")
    for f in files:
        d = np.load(f)
        t = float(d['t']) / 60
        if t < t0 or round(t) % every:
            continue
        T = (d['theta'] + th0) * pi
        a = np.clip((T - 253.15) / 20.0, 0, 1)
        qc, qr = d['qc'], d['qr']
        qg = d['qg'] if 'qg' in d.files else np.zeros_like(qc)
        m = lambda q: float((q * col).sum())
        pg = (qg * col).sum((0, 1))
        zg = (pg * z).sum() / pg.sum() if pg.sum() > 0 else float('nan')
        out = (edge_export(qr + qg, d['u'], d['v'], rho, dz, dx)) * 300 / 1e6
        hi = (qc[:, :, z > 5000] > 1e-5).any(2).mean() * 100
        q0 = qr[:, :, 0]
        rain = (rho[0] * q0 * 36.34 * (rho[0] * q0 / 1000) ** 0.1364 * 3600).max()
        print(f"{t:4.0f} | {m(a * qc):7.1f} {m((1 - a) * qc):6.1f} {m(qr):6.1f} {m(qg):6.1f} | {zg:5.0f} | {out:6.2f}"
              f"                    | {hi:5.1f}%    | {d['theta'][:, :, 0].min():+6.2f} K  | {rain:5.1f} mm/h")


def budget(run, t0, t1):
    f = os.path.join(run, 'micro_budget.npz')
    if not os.path.exists(f):
        print(f"{os.path.basename(run)}: no micro_budget.npz")
        return
    b = np.load(f)
    t = b['times'] / 60
    i0, i1 = np.argmin(abs(t - t0)), np.argmin(abs(t - t1))
    c = b['cumulative_kg'][i1] - b['cumulative_kg'][i0]          # (nz, n_proc)
    P = list(b['processes'])
    tot = c.sum(0) / 1e6
    zbar = (c * b['z'][:, None]).sum(0) / np.where(c.sum(0) > 0, c.sum(0), 1)
    print(f"\n{os.path.basename(run)} process budget {t[i0]:.0f}–{t[i1]:.0f} min [kt, mean height m]:")
    for p, v, zz in zip(P, tot, zbar):
        if v > 0:
            print(f"  {p:14s} {v:8.2f}  {zz:5.0f}" if not p.endswith('_sfc') else f"  {p:14s} {v:8.2f}")
    v = dict(zip(P, tot))
    formed = v['cloud_to_rain'] + v['ice_autoconv'] + v['ice_collect'] + v['riming']
    lost = v['rain_evap'] + v['graupel_subl']
    sfc = v['rain_sfc'] + v['graupel_sfc']
    if formed > 0:
        print(f"  precipitation formed {formed:.2f} kt: evaporated/sublimated {lost / formed * 100:.0f} %"
              f" (rain evap {v['rain_evap'] / formed * 100:.0f} %, graupel subl {v['graupel_subl'] / formed * 100:.0f} %),"
              f" at the ground {sfc / formed * 100:.0f} %")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--t0', type=float, default=60.0)
    ap.add_argument('--every', type=int, default=15)
    ap.add_argument('--budget', type=float, nargs=2, metavar=('T0', 'T1'))
    a = ap.parse_args()
    for r in a.runs:
        summary(r)
    if a.budget:
        for r in a.runs:
            budget(r, *a.budget)
    for r in a.runs:
        print(f"\n{os.path.basename(r)}")
        snapshots(r, a.t0, a.every)


if __name__ == '__main__':
    sys.exit(main())
