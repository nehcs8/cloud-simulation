"""
Test the pressure projection with open lateral boundaries (pressure.py, cfg.BC_X / BC_Y = "open").

For x-open, y-open and both-open, numba and numpy paths, on a stretched grid with a
non-square, odd/even horizontal size and real ρ₀(z):
  - residual divergence ∇·(ρ₀u) after projection ≤ 1e-10 of the initial one
  - net mass flux through the open boundaries is zero afterwards
  - the boundary normal velocities change only by one uniform constant (the flux balance)
  - projecting again changes nothing (idempotent, ≤ 1e-10 of the first correction)
  - a uniform through-flow u = U (already divergence-free) is kept exactly
  - periodic directions keep u[nx] = u[0] / v[:, ny] = v[:, 0]

Usage: python tests/test_pressure_open.py
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config as cfg
cfg.NX, cfg.NY, cfg.DX, cfg.DY = 24, 19, 100.0, 80.0
cfg.GRID_TYPE = "uniform_stretched"
cfg.DZ_UNIFORM, cfg.Z_UNIFORM_TOP, cfg.Z_TOP, cfg.N_STRETCH = 50.0, 1000.0, 2000.0, 6

from grid import Grid
import pressure


def divergence(u, v, w, g, rho0_c):
    """(1/ρ₀) ∇·(ρ₀u) on cell centres, the same discretisation as pressure.project."""
    rho0_w = np.empty(g.nz + 1)
    rho0_w[0], rho0_w[-1] = rho0_c[0], rho0_c[-1]
    rho0_w[1:-1] = 0.5 * (rho0_c[:-1] + rho0_c[1:])
    f = w * rho0_w
    return ((u[1:] - u[:-1]) / g.dx + (v[:, 1:] - v[:, :-1]) / g.dy
            + (f[:, :, 1:] - f[:, :, :-1]) / (g.dz_face * rho0_c))


def net_flux(u, v, g, rho0_c, open_x, open_y):
    wx, wy = rho0_c * g.dz_face * g.dy, rho0_c * g.dz_face * g.dx
    f = 0.0
    if open_x:
        f += ((u[-1] - u[0]) * wx).sum()
    if open_y:
        f += ((v[:, -1] - v[:, 0]) * wy).sum()
    return f


def random_fields(g, rng):
    u = rng.standard_normal((g.nx + 1, g.ny, g.nz)) + 3.0
    v = rng.standard_normal((g.nx, g.ny + 1, g.nz)) - 1.0
    w = rng.standard_normal((g.nx, g.ny, g.nz + 1))
    w[:, :, 0] = w[:, :, -1] = 0.0
    return u, v, w


def check_case(open_x, open_y, use_numba, g, rho0_c, rng, dt=2.0):
    cfg.BC_X = "open" if open_x else "periodic"
    cfg.BC_Y = "open" if open_y else "periodic"
    cfg.USE_NUMBA = use_numba
    label = f"BC_X={cfg.BC_X:8s} BC_Y={cfg.BC_Y:8s} numba={use_numba!s:5s}"

    u, v, w = random_fields(g, rng)
    if not open_x:
        u[-1] = u[0]
    if not open_y:
        v[:, -1] = v[:, 0]
    u0, v0 = u.copy(), v.copy()
    div0 = np.abs(divergence(u, v, w, g, rho0_c)).max()

    pressure.project(u, v, w, g, dt, rho0=rho0_c)
    div1 = np.abs(divergence(u, v, w, g, rho0_c)).max()
    flux = net_flux(u, v, g, rho0_c, open_x, open_y)

    # boundary normal velocity: changed by one uniform constant only
    deltas = []
    if open_x:
        deltas += [u[0] - u0[0], -(u[-1] - u0[-1])]
    if open_y:
        deltas += [v[:, 0] - v0[:, 0], -(v[:, -1] - v0[:, -1])]
    d = np.concatenate([x.ravel() for x in deltas])
    spread = d.max() - d.min()

    # idempotence
    u1, v1, w1 = u.copy(), v.copy(), w.copy()
    pressure.project(u, v, w, g, dt, rho0=rho0_c)
    change = max(np.abs(u - u1).max(), np.abs(v - v1).max(), np.abs(w - w1).max())
    first = max(np.abs(u1 - u0).max(), np.abs(v1 - v0).max())

    print(f"{label}: div {div0:.2e} → {div1:.2e}  net flux {flux:+.1e}  "
          f"boundary Δ spread {spread:.1e} (corr {d.mean():+.3f})  re-project Δ {change:.1e}")
    assert div1 <= 1e-10 * div0, "divergence not removed"
    assert abs(flux) <= 1e-9 * np.abs(u0).sum(), "net boundary flux not zero"
    assert spread <= 1e-12, "boundary normal velocity changed beyond the uniform balance"
    assert change <= 1e-10 * first, "projection not idempotent"
    if not open_x:
        assert np.array_equal(u[-1], u[0])
    if not open_y:
        assert np.array_equal(v[:, -1], v[:, 0])


def check_uniform_throughflow(g, rho0_c):
    cfg.BC_X, cfg.BC_Y, cfg.USE_NUMBA = "open", "open", True
    u = np.full((g.nx + 1, g.ny, g.nz), 5.0)
    v = np.full((g.nx, g.ny + 1, g.nz), -2.0)
    w = np.zeros((g.nx, g.ny, g.nz + 1))
    pressure.project(u, v, w, g, 2.0, rho0=rho0_c)
    err = max(np.abs(u - 5.0).max(), np.abs(v + 2.0).max(), np.abs(w).max())
    print(f"uniform through-flow (5, −2, 0) m/s: max change {err:.1e}")
    assert err < 1e-12


def main():
    g = Grid(terrain=None)
    rho0_c = pressure._constant_theta_rho0(g.z)
    rng = np.random.default_rng(1)
    print(g)
    for use_numba in (True, False):
        for open_x, open_y in ((True, False), (False, True), (True, True)):
            check_case(open_x, open_y, use_numba, g, rho0_c, rng)
    check_uniform_throughflow(g, rho0_c)
    cfg.BC_X = cfg.BC_Y = "periodic"
    print("All open-boundary pressure tests passed.")


if __name__ == "__main__":
    main()
