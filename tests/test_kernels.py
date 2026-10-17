"""
Numba kernels (kernels.py) vs the numpy reference in dynamics.py.

Builds a land_cu state from a convective snapshot (or random fields when no
snapshot is given), then compares, for both paths:
  - WENO-3 scalar advection, variable-K diffusion, K_M, momentum advection
  - saturation adjustment (thermodynamics.diagnose_cloud_water)
  - Kessler microphysics (saturation adjustment, warm rain, sedimentation)
  - the pressure projection (numba Thomas solver vs numpy)
  - the full compute_tendencies() output (cfg.USE_NUMBA toggled)
and times one compute_tendencies() call each way.

Usage: python tests/test_kernels.py [snapshot.npz] [NX]
Success: max relative difference < 1e-10 for every field.
"""

import copy
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config as cfg
from cases import land_cu

cfg.CASE = "land_cu"
land_cu.configure()

snap_path = sys.argv[1] if len(sys.argv) > 1 else None
cfg.NX = cfg.NY = int(sys.argv[2]) if len(sys.argv) > 2 else 64

import dynamics
import halo
import kernels
import pressure
import microphysics
import thermodynamics
from grid import Grid
from state import State

TOL = 1e-10


def _faces_from_centres(uc, vc, wc):
    """C-grid face velocities from cell-centred snapshot fields (periodic x, y; w=0 at lid)."""
    u = np.empty((uc.shape[0] + 1,) + uc.shape[1:])
    u[1:-1] = 0.5 * (uc[:-1] + uc[1:]); u[0] = 0.5 * (uc[-1] + uc[0]); u[-1] = u[0]
    v = np.empty((vc.shape[0], vc.shape[1] + 1, vc.shape[2]))
    v[:, 1:-1] = 0.5 * (vc[:, :-1] + vc[:, 1:]); v[:, 0] = 0.5 * (vc[:, -1] + vc[:, 0]); v[:, -1] = v[:, 0]
    w = np.zeros(wc.shape[:2] + (wc.shape[2] + 1,))
    w[:, :, 1:-1] = 0.5 * (wc[:, :, :-1] + wc[:, :, 1:])
    return u, v, w


def build_state():
    g = Grid(terrain=None)
    s = State(g)
    if snap_path:
        d = {k: np.asarray(v, dtype=np.float64) for k, v in np.load(snap_path).items()}  # snapshots are float32
        assert d["u"].shape == (g.nx, g.ny, g.nz), "snapshot grid does not match NX/land_cu grid"
        s.u, s.v, s.w = _faces_from_centres(d["u"], d["v"], d["w"])
        s.theta, s.qv, s.qc, s.qr = (d[k].copy() for k in ("theta", "qv", "qc", "qr"))
    else:
        rng = np.random.default_rng(1)
        s.u += rng.normal(0, 1.0, s.u.shape); s.u[-1] = s.u[0]
        s.v += rng.normal(0, 1.0, s.v.shape); s.v[:, -1] = s.v[:, 0]
        s.w[:, :, 1:-1] = rng.normal(0, 1.0, s.w[:, :, 1:-1].shape)
        s.theta += rng.normal(0, 0.5, s.theta.shape)
        s.qr = np.abs(rng.normal(0, 1e-4, s.qr.shape))
    s._update_conserved_variables()
    return s, g


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def check(name, a, b, results):
    r = rel(a, b)
    ok = r < TOL
    results.append(ok)
    print(f"  {name:<28} max rel diff {r:9.2e}   {'OK' if ok else 'FAIL'}")


def test_kernels():
    s, g = build_state()
    results = []
    dx, dy, dz_f, dz_c = g.dx, g.dy, g.dz_face, g.dz_cen
    rho0_c = s.rho0
    rho0_w = np.empty(g.nz + 1)
    rho0_w[0], rho0_w[-1] = rho0_c[0], rho0_c[-1]
    rho0_w[1:-1] = 0.5 * (rho0_c[:-1] + rho0_c[1:])
    pc = halo.pad_c                                       # stencils read halo-padded copies

    print(f"Grid {g.nx}x{g.ny}x{g.nz}, {'snapshot ' + snap_path if snap_path else 'random fields'}")
    print("Kernels:")
    for name, phi in (("theta_l", s.theta_l), ("qt", s.qt), ("qr", s.qr)):
        P = pc(phi)
        ref = (dynamics._weno3_x(P, s.u, dx) + dynamics._weno3_y(P, s.v, dy)
               + dynamics._weno3_z(phi, s.w, dz_f, rho0_c, rho0_w))
        check(f"WENO-3 {name}", kernels.weno3_div(P, s.u, s.v, s.w, dx, dy,
                                                  dz_f, rho0_c, rho0_w), ref, results)

    cfg.USE_NUMBA = False
    K_M_ref, K_H = dynamics._smag_K(s, g)
    cfg.USE_NUMBA = True
    K_M_nb, _ = dynamics._smag_K(s, g)
    check("Smagorinsky K_M", K_M_nb, K_M_ref, results)

    P, PK = pc(s.qt), pc(K_H)
    ref = (dynamics._kdiff_x(P, PK, dx) + dynamics._kdiff_y(P, PK, dy)
           + dynamics._kdiff_z(s.qt, K_H, dz_f, dz_c, rho0_c, rho0_w))
    check("K-diffusion qt", kernels.kdiff_xyz(P, PK, dx, dy, dz_f, dz_c, rho0_c, rho0_w),
          ref, results)

    UP, VP, WP = halo.pad_u(s.u), halo.pad_v(s.v), pc(s.w)
    tu, tv, tw = dynamics.advect_momentum(UP, VP, WP, g, rho0_c, rho0_w)
    nu, nv, nw = kernels.momentum_adv(UP, VP, WP, dx, dy, dz_f, dz_c, rho0_c, rho0_w)
    check("momentum adv u", nu, tu, results)
    check("momentum adv v", nv, tv, results)
    check("momentum adv w", nw, tw, results)

    print("Saturation adjustment:")
    pi0, qt_nr = s._pi0(), np.maximum(s.qt - s.qr, 0.0)
    cfg.USE_NUMBA = False
    ref = thermodynamics.diagnose_cloud_water(s.theta_l, qt_nr, s.theta0, pi0)
    cfg.USE_NUMBA = True
    new = thermodynamics.diagnose_cloud_water(s.theta_l, qt_nr, s.theta0, pi0)
    check("diagnose qv", new[0], ref[0], results)
    check("diagnose qc", new[1], ref[1], results)

    print("Microphysics (apply_microphysics, dt = 2 s):")
    out = {}
    for flag in (False, True):
        cfg.USE_NUMBA = flag
        sc = copy.deepcopy(s)
        sc.qr = sc.qr + 2e-4 * (sc.qc > 0)          # some rain inside clouds
        microphysics.apply_microphysics(sc, g, 2.0)
        out[flag] = sc
    for k in ("theta", "qv", "qc", "qr", "surface_precip_flux"):
        check(f"micro {k}", getattr(out[True], k), getattr(out[False], k), results)

    print("Pressure projection:")
    out = {}
    for flag in (False, True):
        cfg.USE_NUMBA = flag
        u, v, w = s.u.copy(), s.v.copy(), s.w.copy()
        phi = pressure.project(u, v, w, g, 2.0, rho0=s.rho0)
        out[flag] = (u, v, w, phi)
    for name, a, b in zip(("u", "v", "w", "phi"), out[True], out[False]):
        check(f"project {name}", a, b, results)

    print("compute_tendencies():")
    cfg.USE_NUMBA = False
    t0 = time.perf_counter(); ref = dynamics.compute_tendencies(s, g); t_np = time.perf_counter() - t0
    cfg.USE_NUMBA = True
    dynamics.compute_tendencies(s, g)                       # JIT warm-up / cache load
    t0 = time.perf_counter(); new = dynamics.compute_tendencies(s, g); t_nb = time.perf_counter() - t0
    for k in ref:
        check(f"tendency {k}", new[k], ref[k], results)
    print(f"Timing: numpy {1e3*t_np:.0f} ms, numba {1e3*t_nb:.0f} ms  "
          f"→ {t_np/t_nb:.1f}× (compute_tendencies only)")
    return all(results)


if __name__ == "__main__":
    ok = test_kernels()
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
