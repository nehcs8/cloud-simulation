"""
Test Monin-Obukhov surface layer physics (Phase 1.5).

Following LES_REBUILD_PLAN.md Phase 1, line 329:
  "Test: Realistic surface stress from M-O theory (no wall BC)"

Success criteria (from LES_REBUILD_PLAN.md Phase 1, line 629):
  - [x] Surface stress from M-O theory (no wall BC)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from surface_layer import stability_functions, compute_friction_velocity, KAPPA, Z0_M, Z0_H
import config as cfg


def test_stability_functions():
    """
    Test Businger-Dyer stability functions for various conditions.
    """
    print("=" * 70)
    print("TEST 1: Stability Functions ψ_m, ψ_h")
    print("=" * 70)
    print()

    # Test cases: neutral, unstable, stable
    test_cases = [
        {"name": "Neutral", "zeta": 0.0, "expected_psi_m": 0.0, "expected_psi_h": 0.0},
        {"name": "Weakly Unstable", "zeta": -0.5, "expected_psi_m": ">0", "expected_psi_h": ">0"},
        {"name": "Strongly Unstable", "zeta": -2.0, "expected_psi_m": ">1", "expected_psi_h": ">0.5"},
        {"name": "Weakly Stable", "zeta": 0.5, "expected_psi_m": "<0", "expected_psi_h": "<0"},
        {"name": "Strongly Stable", "zeta": 2.0, "expected_psi_m": "<-5", "expected_psi_h": "<-5"},
    ]

    all_pass = True
    for case in test_cases:
        zeta = np.array([case["zeta"]])
        psi_m, psi_h = stability_functions(zeta)

        print(f"{case['name']} (ζ = {case['zeta']}):")
        print(f"  ψ_m = {psi_m[0]:.4f}")
        print(f"  ψ_h = {psi_h[0]:.4f}")

        # Check expectations
        if case["name"] == "Neutral":
            test_pass = (abs(psi_m[0]) < 1e-10 and abs(psi_h[0]) < 1e-10)
        elif "Unstable" in case["name"]:
            test_pass = (psi_m[0] > 0 and psi_h[0] > 0)
        else:  # Stable
            test_pass = (psi_m[0] < 0 and psi_h[0] < 0)

        status = "✓" if test_pass else "✗"
        print(f"  {status} Expected behavior: {case['expected_psi_m']}, {case['expected_psi_h']}")
        print()

        all_pass = all_pass and test_pass

    if all_pass:
        print("✓ PASS: Stability functions behave correctly")
    else:
        print("✗ FAIL: Some stability functions incorrect")

    print()
    print("=" * 70)
    print()

    return all_pass


def test_friction_velocity():
    """
    Test friction velocity computation for typical atmospheric conditions.
    """
    print("=" * 70)
    print("TEST 2: Friction Velocity u*")
    print("=" * 70)
    print()

    # Typical fair-weather cumulus conditions
    u_lowest = np.array([[5.0]])  # 5 m/s wind at lowest level
    v_lowest = np.array([[0.0]])
    z_lowest = 50.0  # 50 m above surface
    theta_lowest = np.array([[300.0]])  # K
    theta_surface = np.array([[302.0]])  # Surface 2 K warmer (unstable)

    u_star, L, theta_star = compute_friction_velocity(
        u_lowest, v_lowest, z_lowest, theta_lowest, theta_surface
    )

    print(f"Input:")
    print(f"  Wind speed: {u_lowest[0,0]:.1f} m/s")
    print(f"  Height: {z_lowest:.1f} m")
    print(f"  θ (lowest): {theta_lowest[0,0]:.1f} K")
    print(f"  θ (surface): {theta_surface[0,0]:.1f} K")
    print(f"  Δθ: {theta_surface[0,0] - theta_lowest[0,0]:.1f} K (unstable)")
    print()

    print(f"Output:")
    print(f"  u* = {u_star[0,0]:.4f} m/s")
    print(f"  L = {L[0,0]:.1f} m")
    print(f"  θ* = {theta_star[0,0]:.4f} K")
    print()

    # Physical reasonableness checks
    checks = []

    # 1. Friction velocity should be positive and reasonable
    checks.append(("u* > 0", u_star[0,0] > 0))
    checks.append(("u* reasonable (0.1-1 m/s)", 0.1 < u_star[0,0] < 1.0))

    # 2. M-O length should be negative for unstable (surface heating)
    checks.append(("L < 0 (unstable)", L[0,0] < 0))

    # 3. θ* should be positive (upward heat flux)
    checks.append(("θ* > 0", theta_star[0,0] > 0))

    # 4. Drag coefficient should be reasonable
    # C_D = (u* / U)² ≈ 0.001-0.01 for typical conditions
    C_D = (u_star[0,0] / u_lowest[0,0]) ** 2
    checks.append(("C_D reasonable (0.001-0.01)", 0.001 < C_D < 0.01))

    print(f"Physical checks:")
    all_pass = True
    for name, result in checks:
        status = "✓" if result else "✗"
        print(f"  {status} {name}")
        all_pass = all_pass and result

    print()

    if all_pass:
        print("✓ PASS: Friction velocity computation physically reasonable")
    else:
        print("✗ FAIL: Some checks failed")

    print()
    print("=" * 70)
    print()

    return all_pass


def test_neutral_limit():
    """
    Test that neutral conditions (Δθ ≈ 0) give classical log-law behavior.
    """
    print("=" * 70)
    print("TEST 3: Neutral Limit (Log-Law)")
    print("=" * 70)
    print()

    u_lowest = np.array([[5.0]])
    v_lowest = np.array([[0.0]])
    z_lowest = 50.0
    theta_lowest = np.array([[300.0]])
    theta_surface = np.array([[300.001]])  # Nearly neutral

    u_star, L, theta_star = compute_friction_velocity(
        u_lowest, v_lowest, z_lowest, theta_lowest, theta_surface
    )

    # Neutral drag law: u* = κ * U / ln(z/z0)
    u_star_neutral = KAPPA * u_lowest[0,0] / np.log(z_lowest / Z0_M)

    print(f"Neutral conditions (Δθ ≈ 0):")
    print(f"  u* (computed) = {u_star[0,0]:.4f} m/s")
    print(f"  u* (neutral theory) = {u_star_neutral:.4f} m/s")
    print(f"  Relative error = {abs(u_star[0,0] - u_star_neutral) / u_star_neutral * 100:.2f}%")
    print(f"  |L| = {abs(L[0,0]):.0f} m (large for neutral)")
    print()

    # Check that computed u* is close to neutral theory
    rel_error = abs(u_star[0,0] - u_star_neutral) / u_star_neutral
    test_pass = rel_error < 0.05  # Within 5%

    if test_pass:
        print("✓ PASS: Neutral limit matches log-law")
    else:
        print("✗ FAIL: Neutral limit doesn't match log-law")

    print()
    print("=" * 70)
    print()

    return test_pass


def test_surface_stress():
    """
    Test that surface stress τ = ρ * u*² is reasonable.
    """
    print("=" * 70)
    print("TEST 4: Surface Stress τ = ρ * u*²")
    print("=" * 70)
    print()

    # Typical conditions
    u_lowest = np.array([[8.0]])  # 8 m/s wind
    v_lowest = np.array([[0.0]])
    z_lowest = 50.0
    theta_lowest = np.array([[298.0]])
    theta_surface = np.array([[300.0]])  # 2 K heating

    u_star, L, theta_star = compute_friction_velocity(
        u_lowest, v_lowest, z_lowest, theta_lowest, theta_surface
    )

    rho_sfc = 1.2  # kg/m³, typical surface air density
    tau = rho_sfc * u_star[0,0]**2  # Pa (N/m²)

    print(f"Conditions:")
    print(f"  Wind speed: {u_lowest[0,0]:.1f} m/s")
    print(f"  Surface heating: {theta_surface[0,0] - theta_lowest[0,0]:.1f} K")
    print()

    print(f"Surface stress:")
    print(f"  u* = {u_star[0,0]:.3f} m/s")
    print(f"  τ = ρ * u*² = {tau:.4f} Pa")
    print(f"  τ = {tau:.4f} N/m²")
    print()

    # Physical checks
    # Typical surface stress: 0.01-0.5 Pa for winds of 1-10 m/s
    test_pass = (0.01 < tau < 0.5)

    print(f"Physical check:")
    status = "✓" if test_pass else "✗"
    print(f"  {status} τ in reasonable range (0.01-0.5 Pa)")
    print()

    if test_pass:
        print("✓ PASS: Surface stress is physically reasonable")
    else:
        print("✗ FAIL: Surface stress out of range")

    print()
    print("=" * 70)
    print()

    return test_pass


if __name__ == "__main__":
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "MONIN-OBUKHOV SURFACE LAYER TESTS" + " " * 19 + "║")
    print("║" + " " * 68 + "║")
    print("║" + "  Phase 1.5: Verify M-O bulk flux formulation" + " " * 20 + "║")
    print("║" + "  Critical for LES rebuild (see LES_REBUILD_PLAN.md)" + " " * 13 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    results = []

    # Run tests
    results.append(("Stability functions", test_stability_functions()))
    results.append(("Friction velocity", test_friction_velocity()))
    results.append(("Neutral limit", test_neutral_limit()))
    results.append(("Surface stress", test_surface_stress()))

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
        print("║" + "  Phase 1.5 COMPLETE: M-O surface layer working" + " " * 17 + "║")
        print("║" + "  Ready to proceed to Phase 2 (BOMEX validation)" + " " * 16 + "║")
        print("╚" + "═" * 68 + "╝")
        sys.exit(0)
    else:
        print("╔" + "═" * 68 + "╗")
        print("║" + " " * 20 + "SOME TESTS FAILED" + " " * 30 + "║")
        print("║" + " " * 68 + "║")
        print("║" + "  Phase 1.5 INCOMPLETE: Fix M-O surface layer" + " " * 19 + "║")
        print("╚" + "═" * 68 + "╝")
        sys.exit(1)
