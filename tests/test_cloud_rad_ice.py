"""
Cloud LW radiation (forcing.apply_cloud_radiation, RAD_SCHEME "cloud_lw"), cloud-ice sedimentation
(microphysics._sediment_cloud_ice, ICE_SEDIMENT) and the layered_day "turb" initial state, on the layered_day
grid with 8×8 columns: no-ops when off, top cooling / base warming of a liquid altocumulus slab and a thin
cirrus slab (rates and flux budget), ice falling at the Heymsfield–Donner speed with mass conserved, numba = numpy,
RH patches of the turb variant.

Run: ./venv/bin/python3 tests/test_cloud_rad_ice.py
"""
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import config as cfg
import cases
import microphysics
from forcing import apply_cloud_radiation, cloud_radiation_heating

cfg.CASE = 'layered_day'
cases.get('layered_day').configure()
cfg.NX = cfg.NY = 8

from grid import Grid
from state import State


def fresh(**over):
    saved = {k: getattr(cfg, k) for k in over}
    for k, v in over.items():
        setattr(cfg, k, v)
    g = Grid(terrain=None)
    with contextlib.redirect_stdout(io.StringIO()):
        s = State(g)
    for k, v in saved.items():
        setattr(cfg, k, v)
    return g, s


def slab(g, s, z0, z1, q):
    k = (g.z >= z0) & (g.z <= z1)
    s.qc[:] = 0.0
    s.qc[:, :, k] = q
    return np.nonzero(k)[0]


def test_radiation_off_is_noop():
    g, s = fresh()
    slab(g, s, 4000.0, 4500.0, 3e-4)
    cfg.RAD_SCHEME = None
    th = s.theta.copy()
    apply_cloud_radiation(s, g, 3.0)
    assert np.array_equal(th, s.theta)


def _rates(g, s, ks):
    h = cloud_radiation_heating(s, g)[0, 0] * 3600.0          # K/h
    assert np.all(h[np.setdiff1d(np.arange(g.nz), ks)] == 0.0), "clear air must not be heated"
    return h[ks]


def test_altocumulus_slab():
    g, s = fresh()
    ks = slab(g, s, 4000.0, 4500.0, 3e-4)                      # LWP ≈ 0.3 g/kg · 0.75 · 500 m ≈ 110 g/m²
    h = _rates(g, s, ks)
    print(f"  Ac 4.0–4.5 km, 0.3 g/kg: top {h[-1]:+.2f} K/h, base {h[0]:+.2f} K/h, levels {np.round(h, 2)}")
    assert h[-1] < -1.0 and h[0] > 0.0 and h[-1] > -20.0
    # Flux budget: Σ ρ c_p π Δz dθ/dt = F1 e^0 (1 − e^−Q) − F0 (1 − e^−Q) per level, so the column total is bounded
    pi0 = s._pi0()[0, 0]
    e = (cloud_radiation_heating(s, g)[0, 0] * s.rho0 * g.dz_face * cfg.CP * pi0)[ks].sum()
    assert -cfg.RAD_F0 < e < cfg.RAD_F0, e


def test_cirrus_slab():
    g, s = fresh()
    ks = slab(g, s, 9100.0, 10100.0, 3e-5)                     # IWP ≈ 13 g/m², τ_LW ≈ 0.65
    h = _rates(g, s, ks) * 24.0
    print(f"  Ci 9.1–10.1 km, 0.03 g/kg: top {h[-1]:+.2f} K/day, base {h[0]:+.2f} K/day")
    assert h[-1] < 0.0 < h[0]
    assert abs(h[-1]) < 20.0 and h[0] < 40.0                   # observed cirrus LW rates: a few K/day


def _ice_column(g, s):
    return (s.qc * s.rho0 * g.dz_face).sum(axis=2)[0, 0]


def test_ice_sedimentation():
    for numba in (True, False):
        cfg.USE_NUMBA = numba
        g, s = fresh()
        ks = slab(g, s, 9100.0, 10100.0, 3e-5)
        m0 = _ice_column(g, s)
        zc0 = (s.qc[0, 0] * s.rho0 * g.dz_face * g.z).sum() / m0
        dt, n = 10.0, 60
        for _ in range(n):
            microphysics._sediment_cloud_ice(s, g, dt)
        m1 = _ice_column(g, s)
        zc1 = (s.qc[0, 0] * s.rho0 * g.dz_face * g.z).sum() / m1
        v = (zc0 - zc1) / (n * dt)
        print(f"  ice fall (numba={numba}): mass {m1 / m0:.6f} of initial, centroid falls at {v:.2f} m/s")
        assert abs(m1 / m0 - 1.0) < 1e-6
        assert 0.25 < v < 0.9
        if numba:
            q_numba = s.qc.copy()
        else:
            assert np.allclose(q_numba, s.qc, rtol=1e-10, atol=1e-16)
    cfg.USE_NUMBA = True


def test_liquid_does_not_fall():
    g, s = fresh()
    slab(g, s, 1500.0, 2000.0, 5e-4)                          # warm Cu: α = 1, no ice
    q = s.qc.copy()
    microphysics._sediment_cloud_ice(s, g, 10.0)
    assert np.allclose(q, s.qc, atol=1e-18)


def test_turb_initial_state():
    var = cases.get('layered_day').VARIANTS['turb']
    g, s = fresh(NX=32, NY=32, **{k: v for k, v in var.items() if k != 'LS_FORCING'})   # 8² < patch size
    zs, rhs = np.array(var['LD_RH']).T
    qsat = cases.get('layered_day').qsat_profile(s.theta0, g.z)
    rh = s.qv / qsat
    for z in (4200.0, 9200.0):
        k = np.argmin(abs(g.z - z))
        print(f"  turb RH at {g.z[k]:.0f} m: mean {rh[:, :, k].mean():.3f}, std {rh[:, :, k].std():.3f}, "
              f"max {rh[:, :, k].max():.3f}")
        assert rh[:, :, k].std() > 0.01 and rh[:, :, k].max() <= 0.98 + 1e-9
    assert s.qc.max() == 0.0
    k_dry = np.argmin(abs(g.z - 6500.0))
    assert rh[:, :, k_dry].std() < 1e-9                        # the dry slot is not perturbed
    g2, s2 = fresh(NX=32, NY=32)
    k = np.argmin(abs(g.z - 4200.0))
    assert s2.qv[:, :, k].std() < 1e-12                         # default variant: no patches (round-off only)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f()
            print(f"{name}: ok")
