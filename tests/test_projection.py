"""
Test divergence-free projection on C-grid.

This is the most critical test for Phase 1.1: verify that the pressure
projection removes divergence to machine precision.

Following LES_REBUILD_PLAN.md Phase 1.1, line 214-217:
  "Test: Feed random velocity → project() → measure residual divergence"
  "Verify: run divergence test, check max|∇·u| < 1e-12"
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from grid import Grid
from pressure import project
import config as cfg


def compute_divergence(u, v, w, g: Grid):
    """
    Compute divergence on C-grid.

    u: (nx+1, ny, nz) on x-faces
    v: (nx, ny+1, nz) on y-faces
    w: (nx, ny, nz+1) on z-faces

    Returns: (nx, ny, nz) cell-centered divergence
    """
    dx, dy = g.dx, g.dy
    dz_f = g.dz_face

    # Ensure periodic BCs
    u[-1, :, :] = u[0, :, :]
    v[:, -1, :] = v[:, 0, :]

    # C-grid divergence (exact match to Poisson operator)
    div_u = (u[1:, :, :] - u[:-1, :, :]) / dx
    div_v = (v[:, 1:, :] - v[:, :-1, :]) / dy
    div_w = (w[:, :, 1:] - w[:, :, :-1]) / dz_f[np.newaxis, np.newaxis, :]

    return div_u + div_v + div_w


def test_projection_divergence_free():
    """
    Main test: project random velocity → verify divergence < machine precision.

    Success criteria (from LES_REBUILD_PLAN.md Phase 1, line 626):
      - [x] max|∇·u| < 1e-12 after projection
    """
    print("=" * 70)
    print("PHASE 1.1: C-GRID DIVERGENCE-FREE PROJECTION TEST")
    print("=" * 70)
    print()

    # Create grid (uses config settings)
    g = Grid()

    # Random velocity field (C-grid staggered)
    np.random.seed(42)
    u_init = np.random.randn(g.nx + 1, g.ny, g.nz)
    v_init = np.random.randn(g.nx, g.ny + 1, g.nz)
    w_init = np.random.randn(g.nx, g.ny, g.nz + 1)

    # Enforce rigid lid
    w_init[:, :, 0] = 0.0
    w_init[:, :, -1] = 0.0

    # Compute initial divergence
    div_before = compute_divergence(u_init.copy(), v_init.copy(), w_init.copy(), g)
    max_div_before = np.max(np.abs(div_before))
    mean_div_before = np.mean(np.abs(div_before))

    print(f"Initial divergence (random field):")
    print(f"  max|∇·u| = {max_div_before:.6e}")
    print(f"  mean|∇·u| = {mean_div_before:.6e}")
    print()

    # Project to divergence-free
    dt = 0.1  # arbitrary for this test
    u = u_init.copy()
    v = v_init.copy()
    w = w_init.copy()

    phi = project(u, v, w, g, dt)

    # Compute residual divergence
    div_after = compute_divergence(u, v, w, g)
    max_div_after = np.max(np.abs(div_after))
    mean_div_after = np.mean(np.abs(div_after))
    rms_div_after = np.sqrt(np.mean(div_after**2))

    print(f"Residual divergence (after projection):")
    print(f"  max|∇·u| = {max_div_after:.6e}")
    print(f"  mean|∇·u| = {mean_div_after:.6e}")
    print(f"  RMS|∇·u| = {rms_div_after:.6e}")
    print()

    # Check success criterion
    TOLERANCE = 1e-12
    if max_div_after < TOLERANCE:
        print(f"✓ PASS: max|∇·u| < {TOLERANCE:.0e}")
        print()
        print("C-grid projection is EXACT (within machine precision)!")
        print("Phase 1.1 criterion met.")
        success = True
    else:
        print(f"✗ FAIL: max|∇·u| = {max_div_after:.3e} >= {TOLERANCE:.0e}")
        print()
        print("C-grid projection has excessive divergence!")
        print("Check Poisson solver eigenvalues and gradient stencils.")
        success = False

    print()
    print("=" * 70)

    # Additional diagnostics
    print()
    print("Additional diagnostics:")
    print(f"  Divergence reduction factor: {max_div_before / max_div_after:.1e}")
    print(f"  Pressure field: min={phi.min():.3f}, max={phi.max():.3f}")
    print(f"  Velocity change: Δu_max={np.max(np.abs(u - u_init)):.3e}")
    print()

    return success


def test_periodic_boundaries():
    """
    Test that periodic boundaries are correctly enforced on C-grid.
    """
    print("=" * 70)
    print("ADDITIONAL TEST: Periodic Boundary Conditions")
    print("=" * 70)
    print()

    g = Grid()

    # Create smooth velocity field
    u = np.random.randn(g.nx + 1, g.ny, g.nz)
    v = np.random.randn(g.nx, g.ny + 1, g.nz)
    w = np.zeros((g.nx, g.ny, g.nz + 1))

    # Project
    dt = 0.1
    phi = project(u, v, w, g, dt)

    # Check periodic BCs
    u_periodic = np.allclose(u[0, :, :], u[-1, :, :])
    v_periodic = np.allclose(v[:, 0, :], v[:, -1, :])

    print(f"Periodic BC check:")
    print(f"  u[0] == u[-1]: {u_periodic}")
    print(f"  v[:,0] == v[:,-1]: {v_periodic}")

    if u_periodic and v_periodic:
        print()
        print("✓ PASS: Periodic boundaries correctly enforced")
        success = True
    else:
        print()
        print("✗ FAIL: Periodic boundaries violated")
        success = False

    print()
    print("=" * 70)
    print()

    return success


def test_conservation_of_mass():
    """
    Test that projection conserves total mass (∫∇·u dV = 0 for periodic domain).
    """
    print("=" * 70)
    print("ADDITIONAL TEST: Mass Conservation")
    print("=" * 70)
    print()

    g = Grid()

    # Random velocity field
    np.random.seed(123)
    u = np.random.randn(g.nx + 1, g.ny, g.nz)
    v = np.random.randn(g.nx, g.ny + 1, g.nz)
    w = np.random.randn(g.nx, g.ny, g.nz + 1)
    w[:, :, 0] = 0.0
    w[:, :, -1] = 0.0

    # Project
    dt = 0.1
    phi = project(u, v, w, g, dt)

    # Compute domain-integrated divergence
    div = compute_divergence(u, v, w, g)

    # For periodic domain with rigid lid: ∫∇·u dV = 0
    total_div = np.sum(div) * g.dx * g.dy * g.dz_face.mean()

    print(f"Domain-integrated divergence:")
    print(f"  ∫∇·u dV = {total_div:.6e}")

    # For periodic domain with rigid lid, mass conservation depends on
    # mean w being zero (enforced in pressure.py line 150)
    # Tolerance is less strict than pointwise divergence
    if np.abs(total_div) < 1e-8:
        print()
        print("✓ PASS: Mass conserved (∫∇·u dV ≈ 0)")
        success = True
    else:
        print()
        print("✗ FAIL: Mass not conserved")
        success = False

    print()
    print("=" * 70)
    print()

    return success


if __name__ == "__main__":
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "C-GRID PROJECTION TEST SUITE" + " " * 25 + "║")
    print("║" + " " * 68 + "║")
    print("║" + "  Phase 1.1: Verify divergence-free projection" + " " * 20 + "║")
    print("║" + "  Critical for LES rebuild (see LES_REBUILD_PLAN.md)" + " " * 13 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    results = []

    # Run tests
    results.append(("Divergence-free projection", test_projection_divergence_free()))
    results.append(("Periodic boundaries", test_periodic_boundaries()))
    results.append(("Mass conservation", test_conservation_of_mass()))

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
        print("║" + "  Phase 1.1 COMPLETE: C-grid projection is exact" + " " * 17 + "║")
        print("║" + "  Ready to proceed to Phase 1.2 (RK3 integration)" + " " * 15 + "║")
        print("╚" + "═" * 68 + "╝")
        sys.exit(0)
    else:
        print("╔" + "═" * 68 + "╗")
        print("║" + " " * 20 + "SOME TESTS FAILED" + " " * 30 + "║")
        print("║" + " " * 68 + "║")
        print("║" + "  Phase 1.1 INCOMPLETE: Fix projection before continuing" + " " * 8 + "║")
        print("╚" + "═" * 68 + "╝")
        sys.exit(1)
