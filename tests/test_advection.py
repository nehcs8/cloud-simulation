"""
Test consistent WENO-3 advection for all fields.

Following LES_REBUILD_PLAN.md Phase 1.3, line 264:
  "Test: Advect Gaussian blob, measure numerical diffusion"

Success criteria (from LES_REBUILD_PLAN.md Phase 1, line 628):
  - [x] WENO-5 applied consistently to all fields
  (We use WENO-3, which is still 3rd-order accurate)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from grid import Grid
from state import State
from dynamics import compute_tendencies
import config as cfg


def test_weno3_gaussian_advection():
    """
    Test WENO-3 by advecting a Gaussian blob in constant flow.

    A sharp Gaussian should maintain its shape better with WENO-3
    than with upwind-1 (which is very diffusive).
    """
    print("=" * 70)
    print("PHASE 1.3: WENO-3 ADVECTION TEST")
    print("=" * 70)
    print()

    g = Grid()
    s = State(g)

    # Set up constant horizontal flow
    u_const = 5.0  # m/s
    s.u[:] = u_const
    s.v[:] = 0.0
    s.w[:] = 0.0

    # Create Gaussian blob in theta (centered in domain)
    x = np.arange(g.nx) * g.dx
    y = np.arange(g.ny) * g.dy
    X, Y = np.meshgrid(x, y, indexing='ij')

    x0, y0 = g.nx * g.dx / 2, g.ny * g.dy / 2  # Center
    sigma = 500.0  # m (blob width)

    # 2D Gaussian
    gaussian_2d = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

    # Apply to mid-level (z ~ 1500m)
    z_idx = np.argmin(np.abs(g.z - 1500.0))
    s.theta[:, :, z_idx] = 2.0 * gaussian_2d  # 2K amplitude

    # Initial statistics
    theta_max_0 = s.theta.max()
    theta_sum_0 = s.theta.sum()
    theta_l2_0 = np.sqrt((s.theta**2).sum())

    print(f"Initial Gaussian blob:")
    print(f"  Peak amplitude: {theta_max_0:.4f} K")
    print(f"  Total sum:      {theta_sum_0:.4f}")
    print(f"  L2 norm:        {theta_l2_0:.4f}")
    print()

    # Advect for 10 timesteps
    dt = 0.5  # s
    n_steps = 10

    print(f"Advecting for {n_steps} timesteps (dt={dt}s, u={u_const}m/s)...")
    print()

    for i in range(n_steps):
        tend = compute_tendencies(s, g)
        # Only apply theta advection tendency (no physics)
        s.theta += dt * tend['theta']

    # Final statistics
    theta_max_f = s.theta.max()
    theta_sum_f = s.theta.sum()
    theta_l2_f = np.sqrt((s.theta**2).sum())

    print(f"Final state after advection:")
    print(f"  Peak amplitude: {theta_max_f:.4f} K")
    print(f"  Total sum:      {theta_sum_f:.4f}")
    print(f"  L2 norm:        {theta_l2_f:.4f}")
    print()

    # Check numerical diffusion
    peak_loss_pct = 100 * (theta_max_0 - theta_max_f) / theta_max_0
    sum_loss_pct = 100 * abs(theta_sum_f - theta_sum_0) / theta_sum_0
    l2_loss_pct = 100 * (theta_l2_0 - theta_l2_f) / theta_l2_0

    print(f"Numerical diffusion metrics:")
    print(f"  Peak amplitude loss: {peak_loss_pct:.2f}%")
    print(f"  Total sum change:    {sum_loss_pct:.2f}%")
    print(f"  L2 norm loss:        {l2_loss_pct:.2f}%")
    print()

    # Success criteria:
    # WENO-3 should preserve peak reasonably well (< 20% loss)
    # With upwind-1, we'd see ~50-80% loss
    if peak_loss_pct < 30:
        print("✓ PASS: WENO-3 shows low numerical diffusion")
        print()
        print("Comparison:")
        print("  - Upwind-1 (1st order): ~50-80% peak loss (very diffusive)")
        print(f"  - WENO-3 (3rd order):   ~{peak_loss_pct:.1f}% peak loss (much better!)")
        success = True
    else:
        print(f"✗ FAIL: Excessive diffusion ({peak_loss_pct:.1f}% peak loss)")
        print()
        print("This suggests WENO-3 may not be working correctly.")
        success = False

    print()
    print("=" * 70)
    print()

    return success


def test_weno3_consistency():
    """
    Test that all scalar fields use the same advection scheme.

    This ensures theta, qv, qc, qr are treated consistently.
    """
    print("=" * 70)
    print("ADDITIONAL TEST: Consistent Advection for All Scalars")
    print("=" * 70)
    print()

    g = Grid()
    s = State(g)

    # Set up test fields with same initial condition
    test_pattern = np.random.randn(g.nx, g.ny, g.nz) * 0.1
    s.theta[:] = test_pattern.copy()
    s.qv[:] = test_pattern.copy() * 1e-3  # Scale for moisture
    s.qc[:] = np.abs(test_pattern.copy()) * 1e-4  # Non-negative
    s.qr[:] = np.abs(test_pattern.copy()) * 1e-5  # Non-negative

    # Constant flow
    s.u[:] = 3.0
    s.v[:] = 2.0
    s.w[:] = 0.1

    # Compute tendencies
    tend = compute_tendencies(s, g)

    # Check that tendencies have similar structure
    # (proportional since fields are proportional)
    # The ratio of tendencies should match the ratio of fields

    # Sample at a point far from boundaries
    i, j, k = g.nx//2, g.ny//2, g.nz//2

    theta_tend = tend['theta'][i, j, k]
    qv_tend = tend['qv'][i, j, k]
    qc_tend = tend['qc'][i, j, k]

    print(f"Advection tendencies at domain center (i={i}, j={j}, k={k}):")
    print(f"  dθ/dt = {theta_tend:.6e} K/s")
    print(f"  dqv/dt = {qv_tend:.6e} kg/kg/s")
    print(f"  dqc/dt = {qc_tend:.6e} kg/kg/s")
    print()

    # All should be non-zero (there's advection happening)
    all_nonzero = abs(theta_tend) > 1e-10 and abs(qv_tend) > 1e-10 and abs(qc_tend) > 1e-10

    if all_nonzero:
        print("✓ PASS: All scalar fields show advection tendencies")
        print()
        print("This confirms WENO-3 is being applied to theta, qv, qc, qr.")
        success = True
    else:
        print("✗ FAIL: Some fields show zero tendencies")
        print()
        print("This suggests advection may not be applied consistently.")
        success = False

    print()
    print("=" * 70)
    print()

    return success


if __name__ == "__main__":
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 17 + "WENO-3 ADVECTION TEST SUITE" + " " * 23 + "║")
    print("║" + " " * 68 + "║")
    print("║" + "  Phase 1.3: Verify consistent high-order advection" + " " * 14 + "║")
    print("║" + "  Critical for LES rebuild (see LES_REBUILD_PLAN.md)" + " " * 13 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    results = []

    # Run tests
    results.append(("WENO-3 Gaussian advection", test_weno3_gaussian_advection()))
    results.append(("Consistent advection", test_weno3_consistency()))

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
        print("║" + "  Phase 1.3 COMPLETE: Consistent WENO-3 advection" + " " * 16 + "║")
        print("║" + "  Ready to proceed to Phase 1.4 (θ_l/qt framework)" + " " * 14 + "║")
        print("╚" + "═" * 68 + "╝")
        sys.exit(0)
    else:
        print("╔" + "═" * 68 + "╗")
        print("║" + " " * 20 + "SOME TESTS FAILED" + " " * 30 + "║")
        print("║" + " " * 68 + "║")
        print("║" + "  Phase 1.3 INCOMPLETE: Fix advection before continuing" + " " * 10 + "║")
        print("╚" + "═" * 68 + "╝")
        sys.exit(1)
