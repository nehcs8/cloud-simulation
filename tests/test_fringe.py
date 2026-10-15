"""
Fringe-zone lateral BC (lateral_bc.py) and windowed flux maps.

Checks:
  1. With dt ≫ FRINGE_TAU the fringe end becomes an exact copy of the recycling strip,
     cells outside the fringe are untouched, and u[nx] = u[0] stays periodic.
  2. Fringe strength rises monotonically from 0 to 1 within the zone.
  3. Flux maps in fringe mode: exact domain means, homogeneous outside the test region,
     SHF/LHF uncorrelated, std = SHF_SIGMA inside the test region.
  4. Periodic mode reproduces the plain (unwindowed) maps.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config as cfg
from cases import land_cu


def setup(nx=128, ny=32, dx=100.0):
    cfg.CASE = 'land_cu'
    land_cu.configure()
    cfg.NX, cfg.NY, cfg.DX, cfg.DY = nx, ny, dx, dx
    cfg.LATERAL_BC = 'fringe'
    cfg.FRINGE_WIDTH, cfg.FRINGE_TAU, cfg.RECYCLE_X = 1600.0, 20.0, 4800.0
    cfg.HETERO_BUFFER, cfg.HETERO_TAPER = 1000.0, 500.0
    from grid import Grid
    return Grid()


class FakeState:
    def __init__(self, g, rng):
        self.u = rng.standard_normal((g.nx + 1, g.ny, g.nz))
        self.u[-1] = self.u[0]
        self.v = rng.standard_normal((g.nx, g.ny + 1, g.nz))
        self.w = rng.standard_normal((g.nx, g.ny, g.nz + 1))
        for name in ('theta_l', 'qt', 'qr', 'theta', 'qv', 'qc'):
            setattr(self, name, rng.standard_normal((g.nx, g.ny, g.nz)))


def test_fringe_copy():
    from lateral_bc import apply_fringe, fringe_setup
    g = setup()
    f = fringe_setup(g)
    s = FakeState(g, np.random.default_rng(0))
    before = {n: getattr(s, n).copy() for n in ('u', 'v', 'w', 'theta_l', 'qt', 'qr', 'theta', 'qv', 'qc')}
    apply_fringe(s, g, dt=1e6)                          # full relaxation where λ = 1
    n_f, i_rec = f['n_f'], f['i_rec']
    full = f['lam_c'] > 1.0 - 1e-12                     # second half of the fringe
    for n in ('v', 'w', 'theta_l', 'qt', 'qr', 'theta', 'qv', 'qc'):
        a, b = getattr(s, n), before[n]
        assert np.array_equal(a[:g.nx - n_f], b[:g.nx - n_f]), f"{n} changed outside fringe"
        dst_full = np.arange(g.nx - n_f, g.nx)[full]
        src_full = np.arange(i_rec - n_f, i_rec)[full]
        assert np.allclose(a[dst_full], b[src_full]), f"{n} not copied where λ = 1"
    assert np.array_equal(s.u[-1], s.u[0]), "u[nx] != u[0]"
    assert np.allclose(s.u[g.nx - 1], before['u'][i_rec - 1]), "last u face not recycled"
    print(f"  fringe copy: OK (n_f={n_f}, i_rec={i_rec}, {full.sum()} cells at full strength)")


def test_fringe_weights():
    from lateral_bc import fringe_setup
    g = setup()
    f = fringe_setup(g)
    for lam in (f['lam_c'], f['lam_f']):
        assert lam[0] < 0.05 and abs(lam[-1] - 1.0) < 1e-12
        assert np.all(np.diff(lam) >= -1e-15), "fringe strength not monotonic"
    print(f"  fringe weights: OK (cells {np.round(f['lam_c'][:6], 3)} …)")


def test_flux_maps_fringe():
    g = setup()
    cfg.SHF, cfg.LHF, cfg.SHF_SIGMA, cfg.SHF_CORR = 70.0, 300.0, 0.25, 500.0
    from forcing import build_flux_map
    from lateral_bc import hetero_window
    s, l = build_flux_map(g)
    win = hetero_window(g)
    assert abs(s.mean() - 70.0) < 1e-9 and abs(l.mean() - 300.0) < 1e-9, "domain means changed"
    out = win == 0.0
    assert np.ptp(s[out]) < 1e-9 and np.ptp(l[out]) < 1e-9, "patterns outside test region"
    inside = win == 1.0
    r = np.corrcoef(s[inside].ravel(), l[inside].ravel())[0, 1]
    rel = s[inside].std() / s[inside].mean()
    assert abs(r) < 0.05, f"SHF/LHF correlated in test region: {r:.3f}"
    assert 0.2 < rel < 0.3, f"relative std in test region {rel:.3f}, expected ~0.25"
    print(f"  flux maps (fringe): OK (outside {s[out].flat[0]:.2f}/{l[out].flat[0]:.2f} W/m², "
          f"inside rel std {rel:.3f}, corr {r:+.3f})")


def test_flux_maps_periodic():
    g = setup()
    cfg.LATERAL_BC = 'periodic'
    cfg.SHF, cfg.LHF, cfg.SHF_SIGMA, cfg.SHF_CORR = 70.0, 300.0, 0.25, 500.0
    from forcing import build_flux_map
    s, l = build_flux_map(g)
    assert abs(s.mean() - 70.0) < 1e-9 and abs(l.mean() - 300.0) < 1e-9
    assert abs(s.std() / 70.0 - 0.25) < 1e-3
    assert abs(np.corrcoef(s.ravel(), l.ravel())[0, 1]) < 1e-9
    print("  flux maps (periodic): OK")


def test_flux_stripe():
    g = setup(nx=176, ny=16)
    cfg.SHF, cfg.LHF, cfg.SHF_SIGMA = 70.0, 300.0, 0.0
    cfg.SHF_STRIPE_AMP, cfg.SHF_STRIPE_X0, cfg.SHF_STRIPE_WIDTH, cfg.SHF_STRIPE_TAPER = 35.0, 9000.0, 2000.0, 250.0
    from forcing import build_flux_map
    try:
        s, l = build_flux_map(g)
    finally:
        cfg.SHF_STRIPE_AMP = 0.0
    assert np.ptp(s, axis=1).max() < 1e-9, "stripe not uniform in y"
    plateau = (g.x > 9250.0) & (g.x < 10750.0)
    off = (g.x < 9000.0) | (g.x > 11000.0)
    assert np.allclose(s[plateau], 105.0) and np.allclose(s[off], 70.0)
    assert np.allclose(l, 300.0), "LHF must stay homogeneous"
    # integrated excess = AMP × (WIDTH − TAPER) for cos² tapers
    excess = (s[:, 0] - 70.0).sum() * g.dx
    assert abs(excess - 35.0 * 1750.0) < 35.0 * g.dx, excess
    print(f"  warm stripe: OK (plateau 105 W/m², excess {excess / 35.0:.0f} m × 35 W/m²)")


if __name__ == "__main__":
    print("test_fringe.py")
    test_fringe_weights()
    test_fringe_copy()
    test_flux_maps_fringe()
    test_flux_maps_periodic()
    test_flux_stripe()
    print("ALL PASSED")
