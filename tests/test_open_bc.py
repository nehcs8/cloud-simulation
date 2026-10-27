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
  5. OPEN_INFLOW_REF = "mean": the reference converges to the clear-column mean (a cloudy column and a
     cold-pool column excluded), tapered to the sounding between OPEN_REF_Z1 and OPEN_REF_Z2; the
     ghosts and the relaxation then use it; the default "sounding" leaves it untouched
  6. cell perturbations (OPEN_CPM_AMP > 0): θ offsets within ±amp, constant over cell blocks, only in the
     band next to inflow faces and below OPEN_CPM_ZTOP, applied once per period; off by default

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


def check_mean_reference(g):
    s = State(g)
    s.theta[:] = 0.5 + 0.01 * np.arange(g.nx)[:, None, None]       # warm BL, varying along x
    s.qv[:] = s.qv0 + 1e-3
    s.qc[:] = s.qr[:] = s.qg[:] = 0.0                                # (the default sounding makes cloud)
    s.qc[3, 4, 5] = 1e-3                                             # cloudy column: excluded
    s.theta[7, 8, 0] = -2.0                                          # cold pool: excluded
    s._update_conserved_variables()
    cfg.OPEN_REF_Z1, cfg.OPEN_REF_Z2 = 500.0, 1500.0
    open_bc.update_reference(s, g, 600.0)
    untouched = getattr(s, 'open_ref', None) is None                # default "sounding"
    cfg.OPEN_INFLOW_REF, cfg.OPEN_REF_TAU = "mean", 600.0
    for _ in range(200):
        open_bc.update_reference(s, g, 600.0)
    m = np.ones((g.nx, g.ny), bool); m[3, 4] = m[7, 8] = False
    wz = np.clip((1500.0 - g.z) / 1000.0, 0, 1)
    exp_th = wz * s.theta[m].mean(axis=0)
    exp_qv = s.qv0 + wz * (s.qv[m].mean(axis=0) - s.qv0)
    ref = open_bc.references(s)
    e = max(np.abs(ref['theta'] - exp_th).max(), np.abs(ref['qv'] - exp_qv).max() * 1e3,
            np.abs(ref['theta_l'] - wz * s.theta_l[m].mean(axis=0)).max())
    masks = open_bc.inflow_masks(s)
    P = open_bc.inflow_ghosts(halo.pad_c(s.theta_l), ref['theta_l'], masks)
    Y = slice(H, H + g.ny)
    ok_g = np.allclose(P[0, Y][masks['w']], np.broadcast_to(ref['theta_l'], masks['w'].shape)[masks['w']])
    print(f"mean reference: default untouched {untouched}, |ref − expected| {e:.1e} (θ K, q_v g/kg), "
          f"θ_ref {exp_th[0]:.3f} K at {g.z[0]:.0f} m → {ref['theta'][g.z > 1500].max():.1e} above Z2, ghosts use it {ok_g}")
    cfg.OPEN_INFLOW_REF = "sounding"
    assert untouched and e < 1e-12 and ok_g


def check_cell_perturbations(g):
    s = State(g)
    s.u[:] = 4.0; s.v[:] = 0.0                                       # inflow from the west only (y faces closed)
    s.theta[:] = 0.0
    cfg.OPEN_RELAX_WIDTH = 0
    open_bc.apply_relaxation(s, g, 5.0)
    off = np.all(s.theta == 0.0)
    cfg.OPEN_CPM_AMP, cfg.OPEN_CPM_CELL, cfg.OPEN_CPM_BANDS, cfg.OPEN_CPM_ZTOP = 0.3, 4, 2, 500.0
    cfg.OPEN_CPM_PERIOD = 0.0                                        # auto: 4 × 100 m / |(4, −2)| m/s = 89 s
    open_bc.apply_relaxation(s, g, 5.0)
    th = s.theta.copy()
    kz = g.z < 500.0
    band = th[:8, :, :][:, :, kz]
    ok_amp = np.abs(th).max() <= 0.3 and np.abs(band).min() > 0
    ok_where = np.all(th[8:] == 0) and np.all(th[:, :, ~kz] == 0)
    ok_block = np.all(band[:4, :4] == band[0, 0])
    for _ in range(17):                                              # 85 s < period: unchanged
        open_bc.apply_relaxation(s, g, 5.0)
    ok_wait = np.array_equal(s.theta, th)
    open_bc.apply_relaxation(s, g, 5.0)                              # 90 s: new perturbation added
    ok_new = not np.array_equal(s.theta, th)
    print(f"cell perturbations: off by default {off}, |θ′| ≤ amp and band filled {ok_amp}, only band below z_top "
          f"{ok_where}, constant per block {ok_block}, waits one period {ok_wait}, then adds new {ok_new}")
    cfg.OPEN_CPM_AMP, cfg.OPEN_RELAX_WIDTH = 0.0, 4
    assert off and ok_amp and ok_where and ok_block and ok_wait and ok_new


def main():
    g = Grid(terrain=None)
    rng = np.random.default_rng(5)
    s = make_state(g, rng)
    check_ghosts(g, s)
    check_radiation(g)
    check_budget_flux(g, s)
    check_relaxation(g, s)
    check_mean_reference(g)
    check_cell_perturbations(g)
    cfg.BC_X = cfg.BC_Y = "periodic"
    print("All open-boundary condition tests passed.")


if __name__ == "__main__":
    main()
