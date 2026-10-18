"""
Test the terrain-aware stencils in dynamics.compute_tendencies (stage 2 of docs/PLAN_terrain.md).

Gaussian hill on a small stretched grid, velocities projected with the terrain solver
(zero flow through solid faces); BC periodic/periodic and open/open:
  1. numba and numpy paths agree to round-off (all tendencies)
  2. a uniform scalar (q_t) has zero advective + diffusive tendency in the fluid (up to the
     CG divergence tolerance; with open BCs away from the edges, where inflow brings the sounding)
  3. periodic BCs: Σ ρ₀ (scalar tendency) dV over all cells is zero to round-off — no advective or
     diffusive flux crosses a solid face, and solid cells get no tendency
  4. values stored inside the terrain do not affect the tendencies in fluid cells

Usage: python tests/test_terrain_stencils.py
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config as cfg
cfg.NX, cfg.NY, cfg.DX, cfg.DY = 32, 24, 100.0, 100.0
cfg.GRID_TYPE = "uniform_stretched"
cfg.DZ_UNIFORM, cfg.Z_UNIFORM_TOP, cfg.Z_TOP, cfg.N_STRETCH = 50.0, 1500.0, 3000.0, 8
cfg.TERRAIN_PCG_TOL = 1e-12

from grid import Grid
from state import State
from terrain import Terrain
import dynamics
import pressure


def make_state(g, rng, bx, by):
    cfg.BC_X, cfg.BC_Y = bx, by
    s = State(g)
    s.u[:] = 2.0 + rng.standard_normal(s.u.shape)
    s.v[:] = -1.0 + rng.standard_normal(s.v.shape)
    s.w[:, :, 1:-1] = 0.5 * rng.standard_normal((g.nx, g.ny, g.nz - 1))
    pressure.project(s.u, s.v, s.w, g, 1.0, rho0=s.rho0)
    s.theta_l = rng.standard_normal(s.theta_l.shape)
    s.qt = s.qv0 + 1e-3 * np.abs(rng.standard_normal(s.qt.shape))
    s.qr = 1e-4 * np.abs(rng.standard_normal(s.qr.shape))
    return s


def tendencies(s, g, numba):
    cfg.USE_NUMBA = numba
    return dynamics.compute_tendencies(s, g)


def main():
    g = Grid(terrain=None)
    g.terrain = Terrain.from_gaussian_hill(g, h_max=400.0, sigma=600.0)
    solid, fluid = g.terrain.mask, ~g.terrain.mask
    dV = g.dx * g.dy * g.dz_face
    rng = np.random.default_rng(11)
    print(g, f"— hill 400 m, solid {100 * solid.mean():.1f} %")

    for bx, by in (("periodic", "periodic"), ("open", "open")):
        print(f"BC {bx}/{by}")
        s = make_state(g, rng, bx, by)

        # 1. numba vs numpy
        tn, tp = tendencies(s, g, True), tendencies(s, g, False)
        worst = max(np.abs(tn[k] - tp[k]).max() / max(np.abs(tp[k]).max(), 1e-300) for k in tn)
        print(f"  numba vs numpy: max rel diff {worst:.1e}")
        assert worst < 1e-12

        # 2. uniform scalar
        s2 = make_state(g, np.random.default_rng(12), bx, by)
        s2.qt[:] = 0.012                      # (θ_l also carries −w dθ₀/dz, so use q_t)
        t = tendencies(s2, g, True)['qt']
        inner = fluid.copy()
        if bx == "open":                      # inflow faces bring in the sounding's q_t
            edge = np.zeros_like(fluid); edge[:2] = edge[-2:] = True; edge[:, :2] = edge[:, -2:] = True
            inner &= ~edge
        print(f"  uniform scalar: max |tendency| in fluid{' (away from open edges)' if bx == 'open' else ''} "
              f"{np.abs(t[inner]).max():.1e}")
        assert np.abs(t[inner]).max() < 1e-9

        # 3. conservation (periodic only: open boundaries exchange with the outside)
        if bx == "periodic":
            t = tendencies(s, g, True)
            for key in ("theta_l", "qt", "qr"):
                tot = float((t[key] * s.rho0 * dV).sum())
                scale = float(np.abs(t[key] * s.rho0 * dV).sum())
                print(f"  Σρ₀·tend·dV {key:8s} {tot:+.2e} (rel {abs(tot) / scale:.1e}), "
                      f"max |tend| in solid {np.abs(t[key][solid]).max():.1e}")
                assert abs(tot) < 1e-12 * scale

        # 4. garbage inside the terrain is invisible to the fluid
        t_ref = tendencies(s, g, True)
        s3 = make_state(g, np.random.default_rng(11 if bx == "periodic" else 13), bx, by)
        s3.u, s3.v, s3.w = s.u.copy(), s.v.copy(), s.w.copy()
        s3.theta_l, s3.qt, s3.qr = s.theta_l.copy(), s.qt.copy(), s.qr.copy()
        for f in (s3.theta_l, s3.qt, s3.qr):
            f[solid] = 1e3 * rng.standard_normal(solid.sum())
        t3 = tendencies(s3, g, True)
        diff = max(np.abs(t3[k][fluid] - t_ref[k][fluid]).max() for k in ("theta_l", "qt", "qr"))
        print(f"  random values inside the terrain change fluid tendencies by {diff:.1e}")
        assert diff == 0.0

    cfg.BC_X = cfg.BC_Y = "periodic"
    cfg.USE_NUMBA = True
    print("All terrain stencil tests passed.")


if __name__ == "__main__":
    main()
