"""
Paired analysis of a LATERAL_BC = "fringe" run with flux patches.

Usage: python scripts/analyze_fringe_paired.py RUN_DIR [t0 t1] [--jobs N]

The fetch and the test region get the same inflow (the state at RECYCLE_X), so the field at
x − RECYCLE_X is a twin of the field at x without the patches (corr(w) ≈ 1 at the upstream
end of the test region; see CLAUDE.md). Patch effect = field(x) − field(x − RECYCLE_X),
for x in [RECYCLE_X, Lx − FRINGE_WIDTH), at the same time.

Step 1 reduces every snapshot to 2-D fields and caches them in RUN_DIR/reduced_2d.npz
(cloud mask, cloud-top height, condensed-water path, w at 200/500/1000 m, θ and qv at the
lowest level), so the analysis survives snapshot thinning.
Step 2 prints, for the time mean over [t0, t1]:
  - twin correlation of w(500 m) along x
  - patch effect along x (cover, w′², θ_sfc, qv_sfc, CWP)
  - correlation of the cloud-frequency difference with the SHF map shifted downstream by
    (Lx, Ly), its best lag, and a null distribution from flux maps of other seeds
"""
import ast
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

QC_THR = 1e-5


def _reduce(f):
    d = np.load(f)
    qc, z = d['qc'], d['z']
    zf = d['z_face']
    dz = np.diff(zf)
    cloudy = qc > QC_THR
    mask = cloudy.any(axis=2)
    top = np.where(mask, z[cloudy.shape[2] - 1 - np.argmax(cloudy[:, :, ::-1], axis=2)], 0.0)
    cwp = (qc * dz).sum(axis=2)                          # kg/kg·m (× ρ ≈ 1.1 → kg/m²)
    k = {h: int(np.argmin(np.abs(z - h))) for h in (200.0, 500.0, 1000.0)}
    w = d['w']
    wc = 0.5 * (w[:, :, :-1] + w[:, :, 1:]) if w.shape[2] == len(z) + 1 else w
    return dict(t=float(d['t']), mask=mask, top=top.astype(np.float32), cwp=cwp.astype(np.float32),
                w200=wc[:, :, k[200.0]].astype(np.float32), w500=wc[:, :, k[500.0]].astype(np.float32),
                w1000=wc[:, :, k[1000.0]].astype(np.float32),
                th0=d['theta'][:, :, 0].astype(np.float32), qv0=d['qv'][:, :, 0].astype(np.float32),
                u_ml=d['u'][:, :, z < 500.0].mean(axis=2).astype(np.float32),
                v_ml=d['v'][:, :, z < 500.0].mean(axis=2).astype(np.float32))


def load_reduced(run, jobs):
    path = os.path.join(run, 'reduced_2d.npz')
    files = sorted(glob.glob(os.path.join(run, 'snapshots', 'snap_*.npz')))
    if os.path.exists(path):
        r = dict(np.load(path))
        if len(r['t']) >= len(files):
            return r
    with ProcessPoolExecutor(jobs) as ex:
        out = list(ex.map(_reduce, files, chunksize=4))
    r = {k: np.stack([o[k] for o in out]) for k in out[0]}
    np.savez_compressed(path, **r)
    print(f'Reduced {len(files)} snapshots → {path}')
    return r


def flux_map(eff, seed=None):
    import config as cfg
    for k, v in eff.items():
        if hasattr(cfg, k) and not callable(getattr(cfg, k)):
            setattr(cfg, k, v)
    if seed is not None:
        cfg.RANDOM_SEED = seed
    from grid import Grid
    from forcing import build_flux_map
    return build_flux_map(Grid())[0]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    run = args[0]
    t0 = float(args[1]) if len(args) > 1 else 3600.0
    t1 = float(args[2]) if len(args) > 2 else 1e12
    jobs = int(sys.argv[sys.argv.index('--jobs') + 1]) if '--jobs' in sys.argv else 4

    eff = {}
    for line in open(os.path.join(run, 'effective_config.txt')):
        if ' = ' in line:
            k, v = line.split(' = ', 1)
            try:
                eff[k.strip()] = ast.literal_eval(v.strip())
            except (ValueError, SyntaxError):
                eff[k.strip()] = v.strip()
    nx, dx, dy = eff['NX'], eff['DX'], eff['DY']
    sh = int(round(eff['RECYCLE_X'] / dx))
    i_fr = nx - int(round(eff['FRINGE_WIDTH'] / dx))
    i_h0 = int(round((eff['RECYCLE_X'] + eff['HETERO_BUFFER']) / dx))
    x = (np.arange(nx) + 0.5) * dx

    r = load_reduced(run, jobs)
    sel = (r['t'] >= t0 - 1e-6) & (r['t'] <= t1 + 1e-6)
    n = int(sel.sum())
    print(f'Run: {run}\n{n} snapshots in [{t0:.0f}, {t1:.0f}] s; twin shift {sh * dx / 1000:.1f} km, '
          f'patches {i_h0 * dx / 1000:.1f}–{i_fr * dx / 1000:.1f} km')
    print(f"mean wind in the ML: u {r['u_ml'][sel].mean():.2f}, v {r['v_ml'][sel].mean():.2f} m/s")

    T = slice(sh, i_fr)                       # region with a twin
    Tw = slice(0, i_fr - sh)                  # its twin
    w = r['w500'][sel]
    print('\nTwin correlation of w(500 m) and patch effect along x (time mean):')
    print('  x km   r_twin  Δcover %  Δw′²(500)  Δθ_sfc K  Δqv_sfc g/kg  ΔCWP g/m²  ΔSHF W/m²')
    shf = flux_map(eff)
    nb = (i_fr - sh) // 8
    for b in range(0, i_fr - sh, nb):
        a, c = sh + b, min(sh + b + nb, i_fr)
        p, q = slice(a, c), slice(a - sh, c - sh)
        rt = np.corrcoef(w[:, p].ravel(), w[:, q].ravel())[0, 1]
        dcov = 100 * (r['mask'][sel][:, p].mean() - r['mask'][sel][:, q].mean())
        dw = (w[:, p] ** 2).mean() - (w[:, q] ** 2).mean()
        dth = r['th0'][sel][:, p].mean() - r['th0'][sel][:, q].mean()
        dqv = 1e3 * (r['qv0'][sel][:, p].mean() - r['qv0'][sel][:, q].mean())
        dcwp = 1.1e3 * (r['cwp'][sel][:, p].mean() - r['cwp'][sel][:, q].mean())
        dshf = shf[p].mean() - shf[q].mean()
        print(f'  {x[a:c].mean() / 1000:5.2f}  {rt:6.3f}  {dcov:+7.2f}  {dw:+9.3f}  {dth:+8.3f}  '
              f'{dqv:+11.3f}  {dcwp:+9.2f}  {dshf:+8.1f}')

    # Time-mean difference maps over the full twin-paired region
    D = {}
    for k in ('mask', 'cwp', 'th0', 'top'):
        f = r[k][sel].astype(np.float64)
        D[k] = f[:, T].mean(axis=0) - f[:, Tw].mean(axis=0)
    D['wvar'] = (w[:, T] ** 2).mean(axis=0) - (w[:, Tw] ** 2).mean(axis=0)
    cov_t, cov_w = 100 * r['mask'][sel][:, T].mean(), 100 * r['mask'][sel][:, Tw].mean()
    print(f'\nPaired region {sh * dx / 1000:.1f}–{i_fr * dx / 1000:.1f} km: cover {cov_t:.2f} % vs twin '
          f'{cov_w:.2f} %; std of the Δ cloud-frequency map {100 * D["mask"].std():.2f} %')

    def lagcorr(fld, smap, lx, ly):
        s = np.roll(np.roll(smap - smap.mean(), lx, axis=0), ly, axis=1)[T]
        return np.corrcoef(fld.ravel(), s.ravel())[0, 1]

    lags_x = range(0, int(4000 / dx) + 1, max(1, int(250 / dx)))
    lags_y = range(-int(1500 / dy), int(1500 / dy) + 1, max(1, int(250 / dy)))
    for name, fld in (('Δθ_sfc', D['th0']), ('Δw′²(500)', D['wvar']), ('Δcloud freq', D['mask']),
                      ('ΔCWP', D['cwp'])):
        best = max(((lagcorr(fld, shf, lx, ly), lx, ly) for lx in lags_x for ly in lags_y))
        r0 = lagcorr(fld, shf, 0, 0)
        print(f'{name:12s} vs SHF: r(no lag) {r0:+.2f}; best r {best[0]:+.2f} at shift '
              f'Δx {best[1] * dx / 1000:+.2f} km, Δy {best[2] * dy / 1000:+.2f} km')

    # Null distribution: same statistic against flux maps the run never saw (other seeds)
    null0, nullb = [], []
    for seed in range(100, 120):
        s2 = flux_map(eff, seed)
        null0.append(lagcorr(D['mask'], s2, 0, 0))
        nullb.append(max(lagcorr(D['mask'], s2, lx, ly) for lx in lags_x for ly in lags_y))
    print(f'Null (20 unrelated maps) for Δcloud freq: r(no lag) {np.mean(null0):+.2f} ± {np.std(null0):.2f}; '
          f'best-lag r {np.mean(nullb):+.2f} ± {np.std(nullb):.2f} (max {np.max(nullb):+.2f})')

    out = os.path.join(run, 'paired_maps.npz')
    np.savez_compressed(out, x=x[T], shf=shf[T], **{f'd_{k}': v for k, v in D.items()})
    print(f'Difference maps → {out}')


if __name__ == '__main__':
    main()
