"""
Test the "land_cu" initial sounding (cases/land_cu.py).

Success criteria:
  - No grid point is saturated at t=0: qc = 0 everywhere
  - RH hits the sounding targets (0.93 at 700 m, 0.80 at 1700 m, 0.40 above 1900 m)
  - Perturbations are confined to the lowest 200 m
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config as cfg
cfg.CASE = "land_cu"

from grid import Grid
from state import State
from thermodynamics import saturation_mixing_ratio
from cases import land_cu


def test_land_cu_initial_state():
    g = Grid(terrain=None)
    s = State(g)

    # ── No saturation at t=0 ──────────────────────────────────────────────────
    n_cloudy = int((s.qc > 0.0).sum())
    print(f"Cloudy grid points at t=0: {n_cloudy}  (max qc = {s.qc.max():.2e})")
    assert n_cloudy == 0, f"{n_cloudy} saturated grid points at t=0"

    # ── RH targets on the reference profile ───────────────────────────────────
    z_fine = np.arange(g.z[0], 3000.0, 1.0)
    theta_f, qv_f = land_cu.reference_profiles(z_fine)
    rh_f = qv_f / land_cu.qsat_profile(theta_f, z_fine)
    rh_at = lambda zz: np.interp(zz, z_fine, rh_f)
    for zz, target in [(700.0, 0.93), (1700.0, 0.80), (1900.0, 0.40), (2500.0, 0.40)]:
        print(f"RH({zz:.0f} m) = {rh_at(zz):.4f}  (target {target})")
        assert abs(rh_at(zz) - target) < 2e-3

    # ── θ structure ───────────────────────────────────────────────────────────
    th = lambda zz: float(land_cu.theta_profile(np.array([zz]))[0])
    assert abs(th(350.0) - 300.0) < 1e-9
    assert abs((th(1700.0) - th(700.0)) - 4.0) < 1e-6
    assert abs((th(1900.0) - th(1700.0)) - 3.0) < 1e-6
    assert abs((th(2900.0) - th(1900.0)) - 5.0) < 1e-6

    # ── Perturbations only below 200 m ────────────────────────────────────────
    above = g.z >= 200.0
    assert np.all(s.theta[:, :, above] == 0.0)
    assert np.all(s.qv[:, :, above] == s.qv0[np.newaxis, np.newaxis, above])
    below = ~above
    assert np.abs(s.theta[:, :, below]).max() <= 0.1
    assert np.abs(s.qv[:, :, below] - s.qv0[np.newaxis, np.newaxis, below]).max() <= 1e-4 + 1e-12

    # Max RH in the actual (perturbed) field, from the model's q_sat
    pi0 = s._pi0()
    T = (s.theta + s.theta0[np.newaxis, np.newaxis, :]) * pi0
    rh_max = (s.qv / saturation_mixing_ratio(T, pi0)).max()
    print(f"Max RH in perturbed initial field: {rh_max:.4f}")
    assert rh_max < 1.0

    print("✓ PASS: land_cu sounding is unsaturated everywhere at t=0")


if __name__ == "__main__":
    test_land_cu_initial_state()
