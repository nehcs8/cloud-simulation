"""
Test the open lateral boundary conditions (open_bc.py) with BC_X = BC_Y = "open".

  1. inflow faces put the reference profile into the scalar ghost cells, outflow faces keep
     the zero-gradient (edge) ghosts
  2. radiation: a step in u_n next to the east boundary moves out (u_n tendency of the right
     sign at the boundary face), and a uniform flow equal to the reference has zero
     boundary tendency
  3. relaxation acts only next to inflow faces and pulls toward the reference
  4. boundary_outflow equals −Σ ρ₀ (advective + diffusive tendency) dV of a scalar,
     i.e. the budget term matches what the interior stencils do

Usage: python tests/test_open_bc.py
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config as cfg
cfg.NX, cfg.NY, cfg.DX, cfg.DY = 24, 20, 100.0, 100.0
cfg.GRID_TYPE = "uniform_stretched"
cfg.DZ_UNIFORM, cfg.Z_UNIFORM_TOP, cfg.Z_TOP, cfg.N_STRETCH = 50.0, 1000.0, 2000.0, 6
cfg.BC_X = cfg.BC_Y = "open"
cfg.U_GEO, cfg.V_GEO = 4.0, -2.0

from grid import Grid
from state import State
import halo
import open_bc
import dynamics
import pressure
from halo import H


def make_state(g, rng):
    s = State(g)
    s.u[:] = cfg.U_GEO + rng.standard_normal(s.u.shape)
    s.v[:] = cfg.V_GEO + rng.standard_normal(s.v.shape)
    s.w[:, :, 1:-1] = 0.3 * rng.standard_normal((g.nx, g.ny, g.nz - 1))
    s.theta_l = rng.standard_normal(s.theta_l.shape)
    s.qt = s.qv0 + 1e-3 * np.abs(rng.standard_normal(s.qt.shape))
    s.qr = 1e-4 * np.abs(rng.standard_normal(s.qr.shape))
    pressure.project(s.u, s.v, s.w, g, 1.0, rho0=s.rho0)
    s._update_diagnostic_variables()
    return s


def check_ghosts(g, s):
    masks, ref = open_bc.inflow_masks(s), open_bc.references(s)
    P = open_bc.inflow_ghosts(halo.pad_c(s.theta_l), ref['theta_l'], masks)
    Y = slice(H, H + g.ny)
    west, east = P[0, Y], P[H + g.nx + 1, Y]           # outer ghost columns at x < 0 and x > Lx
    w_in, e_out = masks['w'], ~masks['e']
    ok_in = np.all(west[w_in] == 0.0) and np.array_equal(west[~w_in], s.theta_l[0][~w_in])
    ok_out = np.array_equal(east[e_out], s.theta_l[-1][e_out]) and np.all(east[~e_out] == 0.0)
    print(f"ghosts: west {w_in.sum()} inflow faces → reference {ok_in}; "
          f"east {e_out.sum()} outflow faces → edge value {ok_out}")
    assert ok_in and ok_out and w_in.any() and e_out.any()


def check_radiation(g):
    s = State(g)
    s.u[:] = s.u_g[np.newaxis, np.newaxis, :]
    s.v[:] = cfg.V_GEO
    tu, tv = np.zeros_like(s.u), np.zeros_like(s.v)
    open_bc.radiation_tendency(tu, tv, s, g)
    steady = max(np.abs(tu[[0, -1]]).max(), np.abs(tv[:, [0, -1]]).max())
    s.u[-2] += 1.0                                     # stronger flow just inside the east face
    open_bc.radiation_tendency(tu, tv, s, g)
    rising = tu[-1].min()
    print(f"radiation: uniform reference flow → boundary tendency {steady:.1e}; "
          f"u bump at face nx−1 → east-face tendency {rising:+.3f} m/s² (> 0: bump propagates out)")
    assert steady < 1e-12 and rising > 0.0


def check_relaxation(g, s):
    cfg.OPEN_RELAX_WIDTH, cfg.OPEN_RELAX_TAU = 3, 60.0
    s.theta[:] = 1.0
    masks = open_bc.inflow_masks(s)
    before = s.theta.copy()
    open_bc.apply_relaxation(s, g, 10.0)
    d = before - s.theta                               # > 0 where relaxed toward 0
    west = d[0]
    inside = d[g.nx // 2, g.ny // 2]
    ok = (np.all(west[masks['w']] > 0) and np.all(west[~masks['w'] & ~masks['s'][0][None, :]
                                                       & ~masks['n'][0][None, :]] == 0)
          and np.all(inside == 0))
    print(f"relaxation: west inflow cells relaxed {np.all(west[masks['w']] > 0)}, "
          f"interior untouched {np.all(inside == 0)}")
    assert ok


def check_budget_flux(g, s):
    """Σ ρ₀ (WENO advection + K-diffusion) dV of q_t = −boundary_outflow(q_t)."""
    rho0_c = s.rho0
    rho0_w = np.empty(g.nz + 1)
    rho0_w[0], rho0_w[-1] = rho0_c[0], rho0_c[-1]
    rho0_w[1:-1] = 0.5 * (rho0_c[:-1] + rho0_c[1:])
    masks, ref = open_bc.inflow_masks(s), open_bc.references(s)
    P = open_bc.inflow_ghosts(halo.pad_c(s.qt), ref['qt'], masks)
    K = 5.0 + np.abs(np.random.default_rng(2).standard_normal(s.qt.shape))
    PK = halo.pad_c(K)
    tend = (-dynamics.kernels.weno3_div(P, s.u, s.v, s.w, g.dx, g.dy, g.dz_face, rho0_c, rho0_w)
            + dynamics.kernels.kdiff_xyz(P, PK, g.dx, g.dy, g.dz_face, g.dz_cen, rho0_c, rho0_w))
    dV = g.dx * g.dy * g.dz_face
    total = float((tend * rho0_c * dV).sum())
    out = open_bc.boundary_outflow(P, PK, s, g)
    scale = float(np.abs(tend * rho0_c * dV).sum())
    print(f"budget: Σρ₀·tend·dV {total:+.6e}, −boundary outflow {-out:+.6e} (rel diff {abs(total + out)/scale:.1e})")
    assert abs(total + out) < 1e-12 * scale


def main():
    g = Grid(terrain=None)
    rng = np.random.default_rng(5)
    s = make_state(g, rng)
    check_ghosts(g, s)
    check_radiation(g)
    check_budget_flux(g, s)
    check_relaxation(g, s)
    cfg.BC_X = cfg.BC_Y = "periodic"
    print("All open-boundary condition tests passed.")


if __name__ == "__main__":
    main()
