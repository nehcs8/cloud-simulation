"""
Test the pressure projection with terrain (pressure._project_terrain, stage 1 of docs/PLAN_terrain.md).

Gaussian hill on a small stretched grid; BC periodic/periodic, open/periodic, open/open; numba and numpy:
  - ∇·(ρ₀u) in fluid cells removed to ≤ 1e-7 of its initial value (TERRAIN_PCG_TOL = 1e-8)
  - zero normal velocity on every closed face (no opening; staircase: every face touching a solid cell)
  - run for the staircase and for partial bottom cells (TERRAIN_PARTIAL_CELLS)
  - open boundary faces next to fluid cells keep their u_n up to the uniform flux balance
  - re-projecting changes nothing (≤ 1e-6 of the first correction)
  - a terrain without solid cells reproduces the direct spectral projection (≤ 1e-9)
Prints the CG iteration counts.

Usage: python tests/test_pressure_terrain.py
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config as cfg
cfg.NX, cfg.NY, cfg.DX, cfg.DY = 32, 24, 100.0, 100.0
cfg.GRID_TYPE = "uniform_stretched"
cfg.DZ_UNIFORM, cfg.Z_UNIFORM_TOP, cfg.Z_TOP, cfg.N_STRETCH = 50.0, 1500.0, 3000.0, 8

from grid import Grid
from terrain import Terrain
import pressure

CASES = (("periodic", "periodic"), ("open", "periodic"), ("open", "open"))


def divergence(u, v, w, g, rho0_c):
    """Flux divergence (per full cell volume); with terrain the side-face fluxes use the open fractions."""
    rho0_w = np.empty(g.nz + 1)
    rho0_w[0], rho0_w[-1] = rho0_c[0], rho0_c[-1]
    rho0_w[1:-1] = 0.5 * (rho0_c[:-1] + rho0_c[1:])
    f = w * rho0_w
    if getattr(g, 'terrain', None) is not None:
        t = pressure._terrain_setup(g, rho0_c, rho0_w, cfg.BC_X == "open", cfg.BC_Y == "open")
        u, v = t['ax'] * u, t['ay'] * v
    return ((u[1:] - u[:-1]) / g.dx + (v[:, 1:] - v[:, :-1]) / g.dy
            + (f[:, :, 1:] - f[:, :, :-1]) / (g.dz_face * rho0_c))


def random_velocity(g, rng, bx, by):
    u = rng.standard_normal((g.nx + 1, g.ny, g.nz)) + 3.0
    v = rng.standard_normal((g.nx, g.ny + 1, g.nz)) - 1.0
    w = rng.standard_normal((g.nx, g.ny, g.nz + 1))
    w[:, :, 0] = w[:, :, -1] = 0.0
    if bx == "periodic":
        u[-1] = u[0]
    if by == "periodic":
        v[:, -1] = v[:, 0]
    return u, v, w


def solid_faces(fluid, bx, by):
    """Boolean masks of faces touching at least one solid cell."""
    nx, ny, nz = fluid.shape
    sx = np.zeros((nx + 1, ny, nz), bool)
    sx[1:-1] = ~(fluid[:-1] & fluid[1:])
    sx[0] = ~fluid[0] if bx == "open" else ~(fluid[-1] & fluid[0])
    sx[-1] = ~fluid[-1] if bx == "open" else sx[0]
    sy = np.zeros((nx, ny + 1, nz), bool)
    sy[:, 1:-1] = ~(fluid[:, :-1] & fluid[:, 1:])
    sy[:, 0] = ~fluid[:, 0] if by == "open" else ~(fluid[:, -1] & fluid[:, 0])
    sy[:, -1] = ~fluid[:, -1] if by == "open" else sy[:, 0]
    sz = np.zeros((nx, ny, nz + 1), bool)
    sz[:, :, 1:-1] = ~(fluid[:, :, :-1] & fluid[:, :, 1:])
    return sx, sy, sz


def check_case(g_hill, bx, by, numba, rng, rho0):
    cfg.BC_X, cfg.BC_Y, cfg.USE_NUMBA = bx, by, numba
    g = g_hill
    fluid = ~g.terrain.mask
    u, v, w = random_velocity(g, rng, bx, by)
    sx, sy, sz = solid_faces(fluid, bx, by)
    rho0_w = np.empty(g.nz + 1); rho0_w[0], rho0_w[-1] = rho0[0], rho0[-1]; rho0_w[1:-1] = 0.5 * (rho0[:-1] + rho0[1:])
    t = pressure._terrain_setup(g, rho0, rho0_w, bx == "open", by == "open")
    sx, sy = t['ax'] == 0.0, t['ay'] == 0.0          # closed side faces (partial cells: aperture 0)
    u0, v0 = u.copy(), v.copy()
    d0 = np.abs(divergence(u, v, w, g, rho0)[fluid]).max()
    pressure.project(u, v, w, g, 2.0, rho0=rho0)
    it = pressure.last_terrain_iterations
    d1 = np.abs(divergence(u, v, w, g, rho0)[fluid]).max()
    leak = max(np.abs(u[sx]).max(), np.abs(v[sy]).max(), np.abs(w[sz]).max())
    # open boundary faces next to fluid: changed by one constant only
    deltas = []
    if bx == "open":
        deltas += [(u[0] - u0[0])[fluid[0]], -(u[-1] - u0[-1])[fluid[-1]]]
    if by == "open":
        deltas += [(v[:, 0] - v0[:, 0])[fluid[:, 0]], -(v[:, -1] - v0[:, -1])[fluid[:, -1]]]
    spread = (np.ptp(np.concatenate(deltas)) if deltas else 0.0)
    u1, v1, w1 = u.copy(), v.copy(), w.copy()
    pressure.project(u, v, w, g, 2.0, rho0=rho0)
    it2 = pressure.last_terrain_iterations
    change = max(np.abs(u - u1).max(), np.abs(v - v1).max(), np.abs(w - w1).max())
    first = max(np.abs(u1 - u0).max(), np.abs(v1 - v0).max())
    print(f"BC {bx:8s}/{by:8s} numba={numba!s:5s}: CG {it:3d} it, div {d0:.1e} → {d1:.1e}, "
          f"solid-face |u_n| {leak:.0e}, boundary Δ spread {spread:.0e}, re-project Δ {change:.0e} ({it2} it)")
    assert d1 <= 1e-7 * d0, "divergence not removed"
    assert leak == 0.0, "flow through terrain"
    assert spread <= 1e-12, "open boundary velocities changed beyond the balance"
    assert change <= 1e-6 * first, "projection not idempotent"


def check_no_solid(rng, rho0, g_flat):
    """Terrain object without solid cells → same result as the direct spectral projection."""
    for bx, by in CASES:
        cfg.BC_X, cfg.BC_Y, cfg.USE_NUMBA = bx, by, True
        g_ter = Grid(terrain=None)
        g_ter.terrain = Terrain(np.zeros((g_ter.nx, g_ter.ny)), g_ter)
        u, v, w = random_velocity(g_flat, rng, bx, by)
        a = [x.copy() for x in (u, v, w)]
        pressure.project(*a, g_flat, 2.0, rho0=rho0)
        b = [x.copy() for x in (u, v, w)]
        pressure.project(*b, g_ter, 2.0, rho0=rho0)
        diff = max(np.abs(x - y).max() for x, y in zip(a, b))
        print(f"no solid cells, BC {bx}/{by}: |terrain − direct| {diff:.1e} ({pressure.last_terrain_iterations} it)")
        assert diff < 1e-9


def main():
    g_flat = Grid(terrain=None)
    rho0 = pressure._constant_theta_rho0(g_flat.z)
    rng = np.random.default_rng(7)
    for partial in (False, True):
        cfg.TERRAIN_PARTIAL_CELLS = partial
        g_hill = Grid(terrain=None)
        g_hill.terrain = Terrain.from_gaussian_hill(g_hill, h_max=400.0, sigma=600.0)
        hc = g_hill.terrain.hc
        print(f"{'partial cells' if partial else 'staircase'}: hill 400 m, σ 600 m, solid {100 * (hc == 0).mean():.1f} %, "
              f"partial {((hc > 0) & (hc < 1)).sum()} cells")
        for numba in (True, False):
            for bx, by in CASES:
                check_case(g_hill, bx, by, numba, rng, rho0)
    cfg.TERRAIN_PARTIAL_CELLS = False
    check_no_solid(rng, rho0, g_flat)
    cfg.BC_X = cfg.BC_Y = "periodic"
    cfg.USE_NUMBA = True
    print("All terrain pressure tests passed.")


if __name__ == "__main__":
    main()
