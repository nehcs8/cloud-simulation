"""
Resting-atmosphere test with terrain (stage 4 of docs/PLAN_terrain.md).

Dry, stably stratified (N = 0.01 1/s), windless atmosphere over a 500 m Gaussian hill (σ 1.5 km),
no surface fluxes, no surface layer, no noise, 64² at 100 m, 30 min. With masked-cell terrain the
horizontal pressure gradients are evaluated on level surfaces, so the state must stay at rest:
max |u|, |v|, |w| and |θ′| in the fluid ≤ 1e-6 after 30 min.

Run from the repo root: python tests/test_terrain_rest.py   (~3 min; writes runs/<ts>_..._terrain_rest/,
removed afterwards; summary in outputs/logs/test_terrain_rest.txt)
"""

import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import terrain_setups as ts


def main():
    d = ts.run(1800.0, ['NX=64', 'NY=64', 'U_GEO=0.0', 'V_GEO=0.0', 'SNAPSHOT_EVERY=1800.0',
                        "TERRAIN_TYPE='gaussian_hill'", 'TERRAIN_H_MAX=500.0', 'TERRAIN_SIGMA=1500.0'],
               'terrain_rest')
    snap = np.load(os.path.join(d, 'snapshots', 'snap_0001800.npz'))
    import config as cfg
    from grid import Grid
    from terrain import Terrain
    g = Grid(terrain=None)
    fluid = ~Terrain.from_gaussian_hill(g, h_max=500.0, sigma=1500.0).mask
    res = {k: float(np.abs(snap[k][fluid]).max()) for k in ('u', 'v', 'w', 'theta')}
    lines = [f"resting atmosphere over a 500 m hill, 30 min: max |u| {res['u']:.1e}, |v| {res['v']:.1e}, "
             f"|w| {res['w']:.1e} m/s, |θ′| {res['theta']:.1e} K"]
    ok = max(res.values()) <= 1e-6
    lines.append("PASS" if ok else "FAIL")
    print('\n'.join(lines))
    os.makedirs('outputs/logs', exist_ok=True)
    with open('outputs/logs/test_terrain_rest.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    shutil.rmtree(d)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
