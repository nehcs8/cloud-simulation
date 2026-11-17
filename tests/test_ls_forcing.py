"""
Large-scale forcing (forcing.apply_large_scale_forcing, LS_FORCING): table interpolation in t and z, no-op when
off, upwind vertical advection of the mean θ_l / q_t by w_ls, advective tendencies, and the layered_day
trough lifting its moist layers to saturation (forcing alone, no dynamics, on an 8×8 column set).

Run: ./venv/bin/python3 tests/test_ls_forcing.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import config as cfg
import cases
from forcing import apply_large_scale_forcing, large_scale_profiles

cfg.CASE = 'layered_day'
cases.get('layered_day').configure()
cfg.NX = cfg.NY = 8

from grid import Grid
from state import State


def fresh():
    g = Grid(terrain=None)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        s = State(g)
    return g, s


def test_off_is_noop():
    g, s = fresh()
    cfg.LS_FORCING = None
    thl, qt = s.theta_l.copy(), s.qt.copy()
    apply_large_scale_forcing(s, g, 3.0, 0.0)
    assert np.array_equal(thl, s.theta_l) and np.array_equal(qt, s.qt)
    print("off: no-op")


def test_interpolation():
    g, _ = fresh()
    cfg.LS_FORCING = dict(t=[0.0, 1000.0], z=[0.0, 10000.0], w=[[0.0, 0.0], [0.0, 0.1]])
    w, thl, qt = large_scale_profiles(g, 500.0)
    k = np.argmin(abs(g.z - 5000.0))
    assert abs(w[k] - 0.5 * 0.1 * g.z[k] / 10000.0) < 1e-12, w[k]
    assert np.all(thl == 0) and np.all(qt == 0)
    w2 = large_scale_profiles(g, 5000.0)[0]          # held beyond the table end
    assert abs(w2[k] - 0.1 * g.z[k] / 10000.0) < 1e-12
    print(f"interpolation: w(5 km, 500 s) = {w[k]*100:.3f} cm/s, held after the end")


def test_upwind_advection():
    """Dry interior (2–3.5 km, linear θ₀): Δθ_l = −w dt dθ₀/dz for ascent and descent; tendencies add."""
    for wv in (0.05, -0.05):
        g, s = fresh()
        cfg.LS_FORCING = dict(t=[0.0], z=[0.0, 20000.0], w=[[wv, wv]], thl_adv=[[1e-4, 1e-4]], qt_adv=[[0.0, 0.0]])
        thl0 = s.theta_l.mean(axis=(0, 1)).copy()
        dt = 10.0
        apply_large_scale_forcing(s, g, dt, 0.0)
        k = np.where((g.z > 2000.0) & (g.z < 3500.0))[0]
        dthl = (s.theta_l.mean(axis=(0, 1)) - thl0)[k]
        expect = -wv * dt * np.gradient(s.theta0, g.z)[k] + 1e-4 * dt
        assert np.allclose(dthl, expect, rtol=1e-6, atol=1e-9), (dthl, expect)
        print(f"w = {wv:+.2f} m/s: Δθ_l = {dthl.mean():+.5f} K (expected {expect.mean():+.5f})")


def test_trough_saturates_layers():
    """layered_day trough, forcing only: which layer reaches saturation when (q_c > 1e-6 anywhere in the layer)."""
    g, s = fresh()
    cfg.LS_FORCING = "case"
    dt, first = 10.0, {}
    layers = dict(A=(3800.0, 4700.0), C=(8500.0, 10000.0), dry=(5200.0, 8000.0))
    for n in range(int(6 * 3600 / dt)):
        t = n * dt
        apply_large_scale_forcing(s, g, dt, t)
        for name, (z0, z1) in layers.items():
            k = (g.z > z0) & (g.z < z1)
            if name not in first and s.qc[:, :, k].max() > 1e-6:
                first[name] = t + dt
    qc = s.qc.mean(axis=(0, 1))
    print("trough, forcing only — first saturation: "
          + ", ".join(f"{n} {first[n]/3600:.2f} h" if n in first else f"{n} never" for n in layers))
    for name, (z0, z1) in layers.items():
        k = (g.z > z0) & (g.z < z1)
        print(f"  {name}: q_c at 6 h max {qc[k].max()*1e3:.3f} g/kg")
    assert 'A' in first and 'C' in first and 'dry' not in first
    assert first['A'] < 4 * 3600 and first['C'] < 4 * 3600


if __name__ == '__main__':
    test_off_is_noop()
    test_interpolation()
    test_upwind_advection()
    test_trough_saturates_layers()
    print("all passed")
