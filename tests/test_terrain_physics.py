"""
Test the terrain-aware surface physics (stage 3 of docs/PLAN_terrain.md).

Gaussian hill on a small stretched grid:
  1. surface layer: heat and moisture go only into each column's ground-level fluid cell (k_sfc);
     column-integrated ρ₀ c_p π θ′ gain = SHF·dt and ρ₀ q_v gain = LHF/L_v·dt; stress at the ground level
  2. sedimentation: the sweep ends at k_sfc (cells below untouched), numba = numpy,
     column rain mass lost = surface precipitation × dt
  3. Grid.z_agl: height above ground of the first fluid cell is Δz/2 everywhere

Usage: python tests/test_terrain_physics.py
"""

import sys
import os
import copy

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config as cfg
cfg.NX, cfg.NY, cfg.DX, cfg.DY = 32, 24, 100.0, 100.0
cfg.GRID_TYPE = "uniform_stretched"
cfg.DZ_UNIFORM, cfg.Z_UNIFORM_TOP, cfg.Z_TOP, cfg.N_STRETCH = 50.0, 1500.0, 3000.0, 8
cfg.ENABLE_CLOUD_SHADING = False

from grid import Grid
from state import State
from terrain import Terrain
from surface_layer import apply_surface_layer
import microphysics


def main():
    g = Grid(terrain=None)
    g.terrain = Terrain.from_gaussian_hill(g, h_max=400.0, sigma=600.0)
    kk = g.terrain.k_sfc
    nx, ny = g.nx, g.ny
    I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    print(g, f"— hill 400 m, ground level k_sfc {kk.min()}–{kk.max()}")

    # 3. height above ground
    zagl = g.z_agl()[I, J, kk]
    print(f"z_agl of the first fluid cell: {zagl.min():.1f}–{zagl.max():.1f} m (Δz/2 = {g.dz_face[0]/2:.1f} m)")
    assert np.allclose(zagl, 0.5 * g.dz_face[kk])

    # 1. surface layer
    s = State(g)
    rng = np.random.default_rng(3)
    s.u[:] = 3.0 + 0.5 * rng.standard_normal(s.u.shape)
    s.v[:] = 1.0 + 0.5 * rng.standard_normal(s.v.shape)
    shf = np.full((nx, ny), 150.0); lhf = np.full((nx, ny), 300.0)
    th0, qv0, u0 = s.theta.copy(), s.qv.copy(), s.u.copy()
    dt = 2.0
    apply_surface_layer(s, g, dt, shf, lhf)
    dth, dqv = s.theta - th0, s.qv - qv0
    at_ground = np.zeros_like(dth, bool); at_ground[I, J, kk] = True
    pi0 = s._pi0()[0, 0]
    heat = (s.rho0 * cfg.CP * pi0 * g.dz_face * dth).sum(axis=2)       # J/m² per column
    moist = (s.rho0 * g.dz_face * dqv).sum(axis=2)                      # kg/m²
    print(f"surface layer: θ′ changed outside the ground cells: {np.abs(dth[~at_ground]).max():.1e}; "
          f"column heat / (SHF·dt) {heat.min() / (150 * dt):.6f}–{heat.max() / (150 * dt):.6f}; "
          f"moisture / (LHF/Lv·dt) {moist.min() / (300 / cfg.LV * dt):.6f}–{moist.max() / (300 / cfg.LV * dt):.6f}")
    assert np.abs(dth[~at_ground]).max() == 0.0 and np.abs(dqv[~at_ground]).max() == 0.0
    assert np.allclose(heat, 150 * dt, rtol=1e-4) and np.allclose(moist, 300 / cfg.LV * dt, rtol=1e-4)
    du = s.u - u0
    k_changed = np.argmax(np.abs(du) > 0, axis=2)                       # level of the stress per face
    kx = np.maximum(np.roll(kk, 1, 0), kk)                              # periodic: face i ↔ cells i−1, i
    print(f"surface stress at the faces' ground level: {np.array_equal(k_changed[:-1], kx)}")
    assert np.array_equal(k_changed[:-1], kx)

    # 2. sedimentation
    s = State(g)
    s.qr[:] = 0.0
    s.qr[:, :, 5:15] = 2e-3 * np.abs(rng.standard_normal((nx, ny, 10)))
    s.qr[g.terrain.mask] = 7e-4                                         # marker below the ground
    out = {}
    for nb in (True, False):
        cfg.USE_NUMBA = nb
        sc = copy.deepcopy(s)
        microphysics.sedimentation(sc, g, dt)
        out[nb] = sc
    a, b = out[True], out[False]
    agree = max(np.abs(a.qr - b.qr).max(), np.abs(a.surface_precip_flux - b.surface_precip_flux).max())
    below_untouched = np.array_equal(a.qr[g.terrain.mask], s.qr[g.terrain.mask])
    fluid = ~g.terrain.mask
    m0 = (np.where(fluid, s.qr, 0) * s.rho0 * g.dz_face).sum(axis=2)
    m1 = (np.where(fluid, a.qr, 0) * s.rho0 * g.dz_face).sum(axis=2)
    lost = m0 - m1
    precip = a.surface_precip_flux / 3600.0 * dt                         # kg/m²
    print(f"sedimentation: numba vs numpy {agree:.1e}; cells below ground untouched {below_untouched}; "
          f"column loss − precip·dt max {np.abs(lost - precip).max():.1e} kg/m² (loss max {lost.max():.1e})")
    assert agree < 1e-15 and below_untouched
    assert np.abs(lost - precip).max() < 1e-12

    cfg.USE_NUMBA = True
    print("All terrain physics tests passed.")


if __name__ == "__main__":
    main()
