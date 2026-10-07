"""
Test θ_l/qt conserved variable framework.

Following LES_REBUILD_PLAN.md Phase 1.4, line 298:
  "Test: Verify θ_l, qt conserved to machine precision"

Success criteria (from LES_REBUILD_PLAN.md Phase 1, line 629):
  - [x] θ_l, qt conserved to machine precision
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from grid import Grid
from state import State
from thermodynamics import *
import config as cfg


def test_theta_l_qt_conversion():
    """
    Test conservation of (θ_l, qt).

    The θ_l/qt framework conserves these variables exactly.
    When converting (θ, qv, qc) → (θ_l, qt) → (θ, qv, qc),
    the CONSERVED variables (θ_l, qt) should remain unchanged.
    The diagnostic variables (θ, qv, qc) may differ if the initial
    state wasn't in saturation equilibrium.

    What we test:
    1. θ_l and qt are conserved to machine precision
    2. Recovered state is in saturation equilibrium
    """
    print("=" * 70)
    print("PHASE 1.4: θ_l/qt FRAMEWORK TEST")
    print("=" * 70)
    print()

    g = Grid()

    # Create test fields with realistic atmospheric values
    # Perturbations should be small compared to background
    theta = np.random.randn(g.nx, g.ny, g.nz) * 0.5  # ±0.5 K perturbations
    qv = 0.010 + np.abs(np.random.randn(g.nx, g.ny, g.nz)) * 0.001  # ~10 g/kg ± 1 g/kg
    qc = np.abs(np.random.randn(g.nx, g.ny, g.nz)) * 0.0005  # 0-0.5 g/kg cloud
    qr = np.abs(np.random.randn(g.nx, g.ny, g.nz)) * 0.0001  # 0-0.1 g/kg rain

    # Reference profiles (needed for conversion)
    theta0 = 300.0 + 3.0 * g.z / 1000.0  # Simple linear profile

    # Compute proper Exner function
    pi0_1d = np.ones(g.nz)
    for k in range(1, g.nz):
        dz = g.z[k] - g.z[k-1]
        theta_mid = 0.5 * (theta0[k] + theta0[k-1])
        pi0_1d[k] = pi0_1d[k-1] - cfg.G * dz / (cfg.CP * theta_mid)
    pi0 = pi0_1d[np.newaxis, np.newaxis, :]

    print(f"Initial state:")
    print(f"  θ range: [{theta.min():.3f}, {theta.max():.3f}] K")
    print(f"  qv range: [{qv.min()*1e3:.3f}, {qv.max()*1e3:.3f}] g/kg")
    print(f"  qc range: [{qc.min()*1e3:.3f}, {qc.max()*1e3:.3f}] g/kg")
    print()

    # Forward conversion
    theta_l, qt, qr_out = convert_to_theta_l_qt(theta, qv, qc, qr, pi0)

    print(f"Conserved variables after conversion:")
    print(f"  θ_l range: [{theta_l.min():.3f}, {theta_l.max():.3f}] K")
    print(f"  qt range: [{qt.min()*1e3:.3f}, {qt.max()*1e3:.3f}] g/kg")
    print()

    # Backward conversion
    theta_rec, qv_rec, qc_rec, qr_rec = convert_from_theta_l_qt(
        theta_l, qt, qr_out, theta0, pi0
    )

    # Convert recovered state back to conserved variables
    theta_l_rec, qt_rec, qr_rec2 = convert_to_theta_l_qt(theta_rec, qv_rec, qc_rec, qr_rec, pi0)

    # Test 1: Check that CONSERVED variables are preserved
    theta_l_err = np.max(np.abs(theta_l_rec - theta_l))
    qt_err = np.max(np.abs(qt_rec - qt))

    print(f"TEST 1: Conservation of θ_l and qt")
    print(f"  max|θ_l_roundtrip - θ_l| = {theta_l_err:.6e} K")
    print(f"  max|qt_roundtrip - qt|   = {qt_err:.6e} kg/kg")

    conserved_pass = (theta_l_err < 1e-10 and qt_err < 1e-12)
    if conserved_pass:
        print(f"  ✓ PASS: Conserved variables preserved to machine precision")
    else:
        print(f"  ✗ FAIL: Conserved variables not preserved!")
    print()

    # Test 2: Check saturation equilibrium
    # Recovered state should be at saturation (qv = qs or qv < qs with qc = 0)
    theta_abs = theta_rec + theta0[np.newaxis, np.newaxis, :]
    T = theta_abs * pi0
    T_C = T - 273.15
    e_s = 611.2 * np.exp(17.67 * T_C / (T_C + 243.5))
    p = cfg.P00 * pi0 ** (cfg.CP / cfg.RD)
    qs = cfg.EPS * e_s / np.maximum(p - e_s, 1.0)

    # Check saturation: either qv <= qs (unsaturated) or |qv - qs| < tol (saturated)
    sat_deficit = qv_rec - qs
    max_supersaturation = np.max(sat_deficit)

    print(f"TEST 2: Saturation equilibrium")
    print(f"  max(qv - qs) = {max_supersaturation*1e3:.6f} g/kg")
    print(f"  qc_max = {qc_rec.max()*1e3:.6f} g/kg")

    # Allow small numerical tolerance (0.01 g/kg supersaturation)
    saturation_pass = (max_supersaturation < 1e-5)
    if saturation_pass:
        print(f"  ✓ PASS: Recovered state in saturation equilibrium")
    else:
        print(f"  ✗ FAIL: Recovered state has supersaturation!")
    print()

    success = conserved_pass and saturation_pass

    if success:
        print("✓ OVERALL PASS: θ_l/qt framework working correctly")
    else:
        print("✗ OVERALL FAIL: θ_l/qt framework has errors")

    print()
    print("=" * 70)
    print()

    return success


def test_state_initialization():
    """
    Test that State class initializes θ_l and qt correctly.
    """
    print("=" * 70)
    print("ADDITIONAL TEST: State Initialization with θ_l/qt")
    print("=" * 70)
    print()

    try:
        g = Grid()
        s = State(g)

        # Check that all arrays exist and have correct shape
        checks = [
            ("theta_l", s.theta_l.shape == (g.nx, g.ny, g.nz)),
            ("qt", s.qt.shape == (g.nx, g.ny, g.nz)),
            ("theta (diagnostic)", s.theta.shape == (g.nx, g.ny, g.nz)),
            ("qv (diagnostic)", s.qv.shape == (g.nx, g.ny, g.nz)),
            ("qc (diagnostic)", s.qc.shape == (g.nx, g.ny, g.nz)),
        ]

        print(f"Checking array shapes:")
        all_ok = True
        for name, check in checks:
            status = "✓" if check else "✗"
            print(f"  {status} {name}")
            all_ok = all_ok and check

        print()

        # Check that conserved variables are non-NaN
        has_nan = np.isnan(s.theta_l).any() or np.isnan(s.qt).any()

        if not has_nan:
            print("✓ No NaN values in conserved variables")
        else:
            print("✗ NaN detected in conserved variables!")
            all_ok = False

        print()

        # Check total water conservation: qt = qv + qc + qr
        qt_check = s.qv + s.qc + s.qr
        qt_err = np.max(np.abs(qt_check - s.qt))

        print(f"Total water conservation:")
        print(f"  max|qt - (qv+qc+qr)| = {qt_err:.6e}")

        if qt_err < 1e-10:
            print("  ✓ qt consistent with partition")
        else:
            print("  ✗ qt inconsistent!")
            all_ok = False

        print()

        if all_ok:
            print("✓ PASS: State initialization successful")
            success = True
        else:
            print("✗ FAIL: State initialization problems")
            success = False

    except Exception as e:
        print(f"✗ FAIL: Exception during initialization: {e}")
        import traceback
        traceback.print_exc()
        success = False

    print()
    print("=" * 70)
    print()

    return success


if __name__ == "__main__":
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "θ_l/qt CONSERVED VARIABLES TEST SUITE" + " " * 15 + "║")
    print("║" + " " * 68 + "║")
    print("║" + "  Phase 1.4: Verify moist conserved variable framework" + " " * 11 + "║")
    print("║" + "  Critical for LES rebuild (see LES_REBUILD_PLAN.md)" + " " * 13 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    results = []

    # Run tests
    results.append(("θ_l/qt conversion", test_theta_l_qt_conversion()))
    results.append(("State initialization", test_state_initialization()))

    # Summary
    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print()

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    print()

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("╔" + "═" * 68 + "╗")
        print("║" + " " * 18 + "ALL TESTS PASSED!" + " " * 32 + "║")
        print("║" + " " * 68 + "║")
        print("║" + "  Phase 1.4 COMPLETE: θ_l/qt framework working" + " " * 19 + "║")
        print("║" + "  Ready to proceed to Phase 1.5 (M-O surface layer)" + " " * 13 + "║")
        print("╚" + "═" * 68 + "╝")
        sys.exit(0)
    else:
        print("╔" + "═" * 68 + "╗")
        print("║" + " " * 20 + "SOME TESTS FAILED" + " " * 30 + "║")
        print("║" + " " * 68 + "║")
        print("║" + "  Phase 1.4 INCOMPLETE: Fix θ_l/qt before continuing" + " " * 13 + "║")
        print("╚" + "═" * 68 + "╝")
        sys.exit(1)
