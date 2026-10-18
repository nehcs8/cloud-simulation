"""
Linear mountain-wave test with terrain (stage 4 of docs/PLAN_terrain.md).

Uniform U = 10 m/s, dry, constant N = 0.01 1/s, over a Witch-of-Agnesi ridge h = h₀ a²/(x² + a²),
h₀ = 200 m, a = 5 km (Na/U = 5: mostly hydrostatic; Nh₀/U = 0.2: nearly linear). 2-D: 256 × 4 cells at
200 m, open in x (inflow from the west), periodic in y; 25 m levels to 4 km, stretched to 24 km, sponge
from 12 km; no Coriolis, no surface layer. Compared after 2.5 h (default) with linear hydrostatic theory:

  - wave drag: vertical flux of horizontal momentum M(z) = Σ_x ρ₀ u′ w′ Δx (per unit length of ridge)
    should be height-independent and equal to −(π/4) ρ₀(0) N U h₀²
  - vertical velocity: w = U ∂η/∂x with η = h₀ a (a cos lz − x sin lz)/(x² + a²), l = N/U, scaled by
    √(ρ₀(0)/ρ₀(z)) for the anelastic density decrease; pattern correlation and amplitude ratio for
    |x| < 15 km, 0.5 km < z < 6 km

Pass: |M/M_theory − 1| ≤ 0.25 at 1–6 km, correlation ≥ 0.8.

Staircase terrain acts ~0.6 Δz higher than h (2026-10-18, 2.5 h runs, M/M_theory at 1–6 km):
  Δz 25 m, h₀ 200 m: 1.05–1.21, corr 0.89, amplitude 1.00 (default, passes)
  Δz 50 m, h₀ 200 m: 1.24–1.47, corr 0.73, amplitude 1.34
  Δz 50 m, h₀ 100 m: 1.66–1.92, corr 0.50, amplitude 2.01
  Δz 50 m, h₀ 200 m, sponge from 10 km, τ 300 s: 1.24–1.41 (sponge not the cause)
Run from the repo root: python tests/test_mountain_waves.py   (summary in outputs/logs/test_mountain_waves.txt)
"""

import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import terrain_setups as ts

# Usage: python tests/test_mountain_waves.py [T_END_s] [H0=metres] [TAG=name] [KEY=VALUE ...]
#   extra KEY=VALUE are passed to run_case.py after the defaults (e.g. DZ_UNIFORM=25.0)
_args = sys.argv[1:]
T_END = float(_args.pop(0)) if _args and '=' not in _args[0] else 9000.0
_opts = dict(a.split('=', 1) for a in _args)
U, A = 10.0, 5000.0
H0 = float(_opts.pop('H0', 200.0))
TAG = _opts.pop('TAG', 'mountain_waves')
EXTRA = [f'{k}={v}' for k, v in _opts.items()]


def main():
    d = ts.run(T_END, ['NX=256', 'NY=4', 'DX=200.0', 'DY=200.0', f'U_GEO={U}', 'V_GEO=0.0',
                       'ENABLE_CORIOLIS=False', "BC_X='open'", 'SNAPSHOT_EVERY=900.0',
                       'DZ_UNIFORM=25.0', 'Z_UNIFORM_TOP=4000.0', 'Z_TOP=24000.0', 'N_STRETCH=60',
                       'Z_SPONGE=12000.0', 'TAU_SPONGE=120.0',
                       "TERRAIN_TYPE='agnesi_ridge'", f'TERRAIN_H_MAX={H0}', f'TERRAIN_SIGMA={A}'] + EXTRA,
               TAG)
    import config as cfg
    from grid import Grid
    from state import State
    g = Grid(terrain=None)
    s = State(g)
    rho0, z = s.rho0, g.z
    k1, k2 = np.searchsorted(z, 1000.0), np.searchsorted(z, 2000.0)     # above the 30 m mixed layer
    N = np.sqrt(cfg.G / s.theta0[0] * (s.theta0[k2] - s.theta0[k1]) / (z[k2] - z[k1]))
    x = g.x - g.x.mean()

    lines = [f"[{TAG}{' ' + ' '.join(EXTRA) if EXTRA else ''}] Agnesi ridge h0 {H0:.0f} m, a {A/1e3:.0f} km, U {U:.0f} m/s, N {N:.4f} 1/s, t = {T_END/3600:.1f} h"]
    M_th = -np.pi / 4 * rho0[0] * N * U * H0**2
    Ms = []
    for t in (T_END - 900.0, T_END):
        sn = np.load(os.path.join(d, 'snapshots', f'snap_{t:07.0f}.npz'))
        u, w = sn['u'].mean(axis=1), sn['w'].mean(axis=1)               # (nx, nz), y-averaged
        Ms.append((rho0 * ((u - U) * w).sum(axis=0) * g.dx))
    M = 0.5 * (Ms[0] + Ms[1])
    lines.append(f"theory M = −(π/4) ρ₀ N U h₀² = {M_th:.1f} N/m")
    lines.append("time evolution of M / M_theory at 1, 2, 4, 6 km:")
    for t in np.arange(900.0, T_END + 1, 900.0):
        sn_t = np.load(os.path.join(d, 'snapshots', f'snap_{t:07.0f}.npz'))
        Mt = rho0 * ((sn_t['u'].mean(axis=1) - U) * sn_t['w'].mean(axis=1)).sum(axis=0) * g.dx
        lines.append(f"  t {t/3600:4.2f} h: " + '  '.join(f'{Mt[np.searchsorted(z, zz)] / M_th:5.2f}'
                                                       for zz in (1000, 2000, 4000, 6000)))
    ratios = {}
    for zz in (1000, 2000, 3000, 4000, 6000):
        k = np.searchsorted(z, zz)
        ratios[zz] = M[k] / M_th
        lines.append(f"  z {zz/1e3:3.0f} km: M {M[k]:8.1f} N/m  (ratio {ratios[zz]:.2f})")

    # w pattern vs linear hydrostatic theory (anelastic amplitude growth)
    l = N / U
    X, Z = np.meshgrid(x, z, indexing='ij')
    w_th = U * H0 * A * ((-2 * X) * (A * np.cos(l * Z) - X * np.sin(l * Z)) / (X**2 + A**2)**2
                         + (-np.sin(l * Z)) / (X**2 + A**2))
    w_th *= np.sqrt(rho0[0] / rho0)[None, :]
    region = (np.abs(X) < 15000) & (Z > 500) & (Z < 6000)
    wm = sn['w'].mean(axis=1)
    corr = np.corrcoef(wm[region], w_th[region])[0, 1]
    amp = np.sqrt((wm[region]**2).mean() / (w_th[region]**2).mean())
    lines.append(f"w pattern vs theory (|x| < 15 km, 0.5–6 km): correlation {corr:.3f}, rms amplitude ratio {amp:.2f}, "
                 f"max |w| model {np.abs(wm[region]).max():.3f} theory {np.abs(w_th[region]).max():.3f} m/s")
    ok = all(abs(ratios[zz] - 1) <= 0.25 for zz in (1000, 2000, 3000, 4000, 6000)) and corr >= 0.8
    lines.append("PASS" if ok else "FAIL")
    print('\n'.join(lines))
    os.makedirs('outputs/logs', exist_ok=True)
    with open(f'outputs/logs/test_{TAG}.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    np.savez(f'outputs/logs/test_{TAG}_fields.npz', x=x, z=z, w_model=wm, w_theory=w_th, M=M)
    shutil.rmtree(d)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
