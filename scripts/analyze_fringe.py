"""
Statistics along x for runs with LATERAL_BC = "fringe" (or periodic runs, for comparison).

Usage: python scripts/analyze_fringe.py RUN_DIR [t0 t1] [--shf-map]

Per x-region (fetch [0, RECYCLE_X), test region, fringe) and as profiles along x
(averaged over y and the snapshots in [t0, t1]):
  cloud cover (column qc > 1e-5), w variance at ~500 m, mean u and θ in the mixed layer,
  qv in the mixed layer, LWP.
The fetch and test region should agree in a homogeneous run; a jump at x = 0 (just
downstream of the fringe) relative to x = RECYCLE_X shows a bad inflow.
With --shf-map, the flux map is rebuilt from the run's effective_config.txt and the time-mean
cloud frequency is correlated with SHF inside the test region.
"""
import ast
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

args = [a for a in sys.argv[1:] if not a.startswith('--')]
run = args[0]
t0 = float(args[1]) if len(args) > 1 else 3600.0
t1 = float(args[2]) if len(args) > 2 else 1e12

eff = {}
for line in open(os.path.join(run, 'effective_config.txt')):
    if ' = ' in line:
        k, v = line.split(' = ', 1)
        try:
            eff[k.strip()] = ast.literal_eval(v.strip())
        except (ValueError, SyntaxError):
            eff[k.strip()] = v.strip()
nx, dx = eff['NX'], eff['DX']
Lx = nx * dx
fringe = eff.get('LATERAL_BC', 'periodic') == 'fringe'
x = (np.arange(nx) + 0.5) * dx
if fringe:
    x_rec, x_fr = eff['RECYCLE_X'], Lx - eff['FRINGE_WIDTH']
    x_h0 = x_rec + eff['HETERO_BUFFER']
    regions = {'fetch': x < x_rec, 'test': (x >= x_h0) & (x < x_fr), 'fringe': x >= x_fr}
else:
    regions = {'domain': np.ones(nx, bool)}

files = [f for f in sorted(glob.glob(os.path.join(run, 'snapshots', 'snap_*.npz')))
         if t0 - 1e-6 <= int(os.path.basename(f)[5:12]) <= t1 + 1e-6]
if not files:
    sys.exit(f'No snapshots in [{t0}, {t1}] s')

z = np.load(files[0])['z']
ml = z < 500.0
k500 = int(np.argmin(np.abs(z - 500.0)))
acc = {k: np.zeros(nx) for k in ('cover', 'wvar', 'u', 'theta', 'qv', 'lwp')}
freq2d = np.zeros((nx, eff['NY']))
for f in files:
    d = np.load(f)
    cm = (d['qc'] > 1e-5).any(axis=2)
    freq2d += cm
    acc['cover'] += cm.mean(axis=1)
    w = d['w'][:, :, k500]
    acc['wvar'] += ((w - w.mean()) ** 2).mean(axis=1)
    acc['u'] += d['u'][:, :, ml].mean(axis=(1, 2))
    acc['theta'] += d['theta'][:, :, ml].mean(axis=(1, 2))
    acc['qv'] += 1e3 * d['qv'][:, :, ml].mean(axis=(1, 2))
    acc['lwp'] += 1e3 * (d['qc'] * np.gradient(np.append(z, 2 * z[-1] - z[-2]))[:-1]).sum(axis=2).mean(axis=1)
for k in acc:
    acc[k] /= len(files)
freq2d /= len(files)

print(f'Run: {run}')
print(f'{len(files)} snapshots in [{t0:.0f}, {t1:.0f}] s; Lx = {Lx/1000:.1f} km, '
      f'{"fringe: recycling at %.1f km, patches from %.1f km, fringe from %.1f km" % (x_rec/1e3, x_h0/1e3, x_fr/1e3) if fringe else "periodic"}')
print(f"\n{'region':8s} {'cover %':>8s} {'w′² 500m':>9s} {'u_ml':>6s} {'θ′_ml':>7s} {'qv_ml':>7s} {'LWP g/m²':>9s}")
for name, m in regions.items():
    print(f"{name:8s} {100*acc['cover'][m].mean():8.1f} {acc['wvar'][m].mean():9.3f} {acc['u'][m].mean():6.2f} "
          f"{acc['theta'][m].mean():7.3f} {acc['qv'][m].mean():7.3f} {acc['lwp'][m].mean():9.2f}")

nb = 16
print(f"\nProfiles along x ({nb} bins of {Lx/nb/1000:.2f} km):")
print(f"{'x km':>6s} {'cover %':>8s} {'w′²':>7s} {'u_ml':>6s} {'θ′_ml':>7s} {'qv_ml':>7s}")
for b in range(nb):
    m = slice(b * nx // nb, (b + 1) * nx // nb)
    tag = ''
    if fringe:
        xc = x[m].mean()
        tag = ' fetch' if xc < x_rec else (' fringe' if xc >= x_fr else ' test')
    print(f"{x[m].mean()/1000:6.2f} {100*acc['cover'][m].mean():8.1f} {acc['wvar'][m].mean():7.3f} "
          f"{acc['u'][m].mean():6.2f} {acc['theta'][m].mean():7.3f} {acc['qv'][m].mean():7.3f}{tag}")

if '--shf-map' in sys.argv:
    import config as cfg
    for k, v in eff.items():
        if hasattr(cfg, k) and not callable(getattr(cfg, k)):
            setattr(cfg, k, v)
    from grid import Grid
    from forcing import build_flux_map
    shf, _ = build_flux_map(Grid())
    m = regions.get('test', regions.get('domain'))
    s, fr = shf[m].ravel(), freq2d[m].ravel()
    hot, cold = s > np.percentile(s, 75), s < np.percentile(s, 25)
    print(f"\nCloud frequency vs SHF ({'test region' if fringe else 'domain'}): r = {np.corrcoef(fr, s)[0, 1]:+.2f}; "
          f"warmest 25 % {100*fr[hot].mean():.1f} %, coldest 25 % {100*fr[cold].mean():.1f} %")
