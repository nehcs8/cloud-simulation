"""
Test the halo-padded stencils (halo.py, dynamics.compute_tendencies) with open lateral boundaries.

For BC_X / BC_Y in {open/periodic, periodic/open, open/open}:
  1. numba and numpy paths agree to round-off (all tendencies and K_M)
  2. a uniform scalar in a divergence-free flow gets zero advective tendency
  3. flux form: Σ ρ₀ (advective tendency) dV = −(net outward boundary flux of ρ₀ u φ),
     i.e. the scheme conserves the scalar up to what crosses the boundary faces
  4. a blob far from the boundaries in a flow that is uniform near them gives the same
     tendencies with open and periodic boundaries (the halo only acts at the edges)
Also: halo.pad_* reproduce np.roll / np.pad('wrap') values for periodic directions.

Usage: python tests/test_halo.py
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config as cfg
cfg.NX, cfg.NY, cfg.DX, cfg.DY = 28, 22, 100.0, 100.0
cfg.GRID_TYPE = "uniform_stretched"
cfg.DZ_UNIFORM, cfg.Z_UNIFORM_TOP, cfg.Z_TOP, cfg.N_STRETCH = 50.0, 1000.0, 2000.0, 6
cfg.ENABLE_CORIOLIS = True

from grid import Grid
from state import State
import halo
import dynamics
import pressure

CASES = (("open", "periodic"), ("periodic", "open"), ("open", "open"))


def set_bc(bx, by):
    cfg.BC_X, cfg.BC_Y = bx, by


def random_state(g, rng, blob_only=False):
    """State with smooth random fields; velocities projected to ∇·(ρ₀u) = 0."""
    s = State(g)
    nx, ny, nz = g.nx, g.ny, g.nz
    if blob_only:
        # uniform flow plus a localised perturbation well inside the domain
        x = (np.arange(nx) + 0.5) / nx - 0.5
        y = (np.arange(ny) + 0.5) / ny - 0.5
        bump = np.exp(-(x[:, None]**2 + y[None, :]**2) / 0.004)[:, :, None] * np.ones(nz)
        s.u[:] = 3.0
        s.v[:] = -1.0
        s.w[:] = 0.0
        s.w[:, :, 1:-1] = 0.5 * bump[:, :, 1:]
        s.theta_l[:] = 0.5 * bump                              # no background noise: uniform edges
        s.qt[:] = s.qv0 + 1e-3 * bump
        s.qr[:] = 1e-4 * bump
        return s
    s.u += rng.standard_normal(s.u.shape) + 2.0
    s.v += rng.standard_normal(s.v.shape)
    s.w[:, :, 1:-1] = rng.standard_normal((nx, ny, nz - 1))
    s.theta_l += rng.standard_normal(s.theta_l.shape)
    s.qt = np.abs(s.qt + 1e-3 * rng.standard_normal(s.qt.shape))
    s.qr = 1e-4 * np.abs(rng.standard_normal(s.qr.shape))
    pressure.project(s.u, s.v, s.w, g, 1.0, rho0=s.rho0)
    s._update_diagnostic_variables()
    return s


def check_pad_periodic(g, rng):
    set_bc("periodic", "periodic")
    a = rng.standard_normal((g.nx, g.ny, 3))
    P = halo.pad_c(a)
    ref = np.pad(a, ((2, 2), (2, 2), (0, 0)), mode='wrap')
    u = rng.standard_normal((g.nx + 1, g.ny, 3)); u[-1] = u[0]
    PU = halo.pad_u(u)
    refu = np.pad(u[:-1], ((2, 3), (2, 2), (0, 0)), mode='wrap')
    print(f"periodic pad_c == np.pad wrap: {np.array_equal(P, ref)}, pad_u == wrap of unique faces: "
          f"{np.array_equal(PU, refu)}")
    assert np.array_equal(P, ref) and np.array_equal(PU, refu)


def check_numba_numpy(g, rng, bx, by):
    set_bc(bx, by)
    s = random_state(g, rng)
    out = {}
    for nb in (True, False):
        cfg.USE_NUMBA = nb
        t = dynamics.compute_tendencies(s, g)
        t['KM'] = dynamics._smag_K(s, g)[0]
        out[nb] = t
    worst = max(np.abs(out[True][k] - out[False][k]).max() / max(np.abs(out[False][k]).max(), 1e-300)
                for k in out[True])
    print(f"  numba vs numpy: max rel diff {worst:.1e}")
    assert worst < 1e-12
    cfg.USE_NUMBA = True


def adv_only(s, g, phi):
    """Advective tendency −∇·(uφ) (WENO-3, numba) of one scalar."""
    rho0_c = s.rho0
    rho0_w = np.empty(g.nz + 1)
    rho0_w[0], rho0_w[-1] = rho0_c[0], rho0_c[-1]
    rho0_w[1:-1] = 0.5 * (rho0_c[:-1] + rho0_c[1:])
    return -dynamics.kernels.weno3_div(halo.pad_c(phi), s.u, s.v, s.w, g.dx, g.dy,
                                       g.dz_face, rho0_c, rho0_w), rho0_w


def boundary_flux(s, g, phi, bx, by):
    """Net outward flux Σ ρ₀ u_n φ_face dA through the open boundaries (WENO face value)."""
    P = halo.pad_c(phi)
    eps = lambda c: dynamics._weno_eps(c)
    f = 0.0
    dA_x = g.dy * g.dz_face * s.rho0          # (nz,)
    dA_y = g.dx * g.dz_face * s.rho0
    Hh = halo.H
    Y, X = slice(Hh, Hh + g.ny), slice(Hh, Hh + g.nx)
    if bx == "open":
        for m, sign in ((0, -1.0), (g.nx, 1.0)):               # face m between cells m−1 and m
            c = P[Hh + m - 1, Y]
            F = dynamics._weno3_faces(P[Hh + m - 2, Y], c, P[Hh + m, Y], P[Hh + m + 1, Y],
                                      s.u[m], eps(c))
            f += sign * (F * dA_x).sum()
    if by == "open":
        for m, sign in ((0, -1.0), (g.ny, 1.0)):
            c = P[X, Hh + m - 1]
            F = dynamics._weno3_faces(P[X, Hh + m - 2], c, P[X, Hh + m], P[X, Hh + m + 1],
                                      s.v[:, m], eps(c))
            f += sign * (F * dA_y).sum()
    return f


def check_uniform_and_conservation(g, rng, bx, by):
    set_bc(bx, by)
    s = random_state(g, rng)
    dV = g.dx * g.dy * g.dz_face
    # uniform scalar, divergence-free flow → zero tendency
    t_u, _ = adv_only(s, g, np.full((g.nx, g.ny, g.nz), 300.0))
    print(f"  uniform scalar: max |tendency| {np.abs(t_u).max():.1e}")
    assert np.abs(t_u).max() < 1e-9
    # flux form: Σ ρ₀ tend dV = −boundary outflow
    phi = s.theta_l + 300.0
    t, _ = adv_only(s, g, phi)
    total = (t * s.rho0 * dV).sum()
    fb = boundary_flux(s, g, phi, bx, by)
    scale = np.abs(t * s.rho0 * dV).sum()
    print(f"  Σρ₀·tend·dV {total:+.6e}  −boundary flux {-fb:+.6e}  (rel diff {abs(total + fb)/scale:.1e})")
    assert abs(total + fb) < 1e-12 * scale


def check_blob_interior(g, rng):
    res = {}
    for bx, by in (("periodic", "periodic"), ("open", "open")):
        set_bc(bx, by)
        s = random_state(g, rng, blob_only=True)
        res[bx] = dynamics.compute_tendencies(s, g)
    worst = max(np.abs(res["open"][k] - res["periodic"][k]).max() for k in res["open"])
    print(f"blob far from the edges: open vs periodic tendencies max |diff| {worst:.1e}")
    assert worst < 1e-10


def main():
    g = Grid(terrain=None)
    rng = np.random.default_rng(3)
    print(g)
    check_pad_periodic(g, rng)
    for bx, by in CASES:
        print(f"BC_X={bx}, BC_Y={by}")
        check_numba_numpy(g, rng, bx, by)
        check_uniform_and_conservation(g, rng, bx, by)
    check_blob_interior(g, rng)
    set_bc("periodic", "periodic")
    print("All halo tests passed.")


if __name__ == "__main__":
    main()
