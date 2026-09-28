#!/usr/bin/env python3
"""
Test to demonstrate that the current pressure projection doesn't remove divergence.

The problem:
- u, v are cell-centered with 2Δx centered differences
- Poisson solver uses eigenvalues for 1Δx differences
- These don't match → projection can't correct all divergence modes

Expected result BEFORE C-grid fix:
- max|div| after projection: ~50% of initial (NOT near zero!)

Expected result AFTER C-grid fix:
- max|div| after projection: < 1e-12 (machine precision)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from grid import Grid
from pressure import project
import config as cfg

def compute_divergence(u, v, w, g):
    """
    Compute divergence using the current (cell-centered u,v) operators.
    This is the same divergence that project() uses.
    """
    dx, dy = g.dx, g.dy
    dz_f = g.dz_face

    div_u = (np.roll(u, -1, 0) - np.roll(u, 1, 0)) / (2.0 * dx)
    div_v = (np.roll(v, -1, 1) - np.roll(v, 1, 1)) / (2.0 * dy)
    div_w = (w[:, :, 1:] - w[:, :, :-1]) / dz_f[np.newaxis, np.newaxis, :]

    return div_u + div_v + div_w

def test_projection_divergence_removal():
    """
    Test that projection removes divergence.

    BEFORE C-grid fix, this test will FAIL because:
    - Divergence and gradient operators use 2Δx stencils
    - Poisson solver uses 1Δx eigenvalues
    - Mismatch → can only remove ~50% of divergence
    """
    print("\n" + "="*70)
    print("PRESSURE PROJECTION DIVERGENCE TEST")
    print("="*70 + "\n")

    # Create a small grid for testing
    original_nx, original_ny, original_nz = cfg.NX, cfg.NY, cfg.NZ
    cfg.NX, cfg.NY, cfg.NZ = 32, 32, 20  # Small test grid

    try:
        g = Grid()

        # Generate random divergent velocity field
        print("Generating random velocity field...")
        np.random.seed(42)
        u = np.random.randn(g.nx, g.ny, g.nz) * 10.0  # m/s
        v = np.random.randn(g.nx, g.ny, g.nz) * 10.0
        w = np.random.randn(g.nx, g.ny, g.nz + 1) * 5.0
        w[:, :, 0] = 0.0   # Bottom boundary
        w[:, :, -1] = 0.0  # Top boundary

        # Compute initial divergence
        div_initial = compute_divergence(u, v, w, g)
        max_div_initial = np.abs(div_initial).max()

        print(f"Initial divergence:")
        print(f"  max|div| = {max_div_initial:.6e} s⁻¹")
        print(f"  mean|div| = {np.abs(div_initial).mean():.6e} s⁻¹\n")

        # Apply pressure projection
        print("Applying pressure projection...")
        dt = 1.0  # Arbitrary for this test
        phi = project(u, v, w, g, dt)

        # Compute divergence after projection
        div_final = compute_divergence(u, v, w, g)
        max_div_final = np.abs(div_final).max()

        print(f"\nFinal divergence:")
        print(f"  max|div| = {max_div_final:.6e} s⁻¹")
        print(f"  mean|div| = {np.abs(div_final).mean():.6e} s⁻¹")

        # Compute reduction
        reduction = max_div_final / max_div_initial
        print(f"\nDivergence reduction:")
        print(f"  Remaining: {reduction*100:.1f}%")
        print(f"  Removed: {(1-reduction)*100:.1f}%\n")

        # Check result
        print("="*70)
        print("TEST RESULT:")
        print("="*70)

        if max_div_final < 1e-10:
            print("✅ PASS: Divergence removed to machine precision")
            print("   (C-grid implementation working correctly!)")
            return True
        elif reduction > 0.3:
            print("❌ FAIL: Projection only removed ~{:.0f}% of divergence".format((1-reduction)*100))
            print("   Problem: Cell-centered u,v with mismatched Poisson eigenvalues")
            print("   Solution: Implement C-grid staggering (Phase 1.1)")
            print(f"\n   Expected after C-grid fix: max|div| < 1e-12")
            print(f"   Current: max|div| = {max_div_final:.6e} (too large!)")
            return False
        else:
            print("⚠️  PARTIAL: Projection removed {:.0f}% (better than baseline)".format((1-reduction)*100))
            print("   Still needs improvement to reach machine precision")
            return False

    finally:
        # Restore original config
        cfg.NX, cfg.NY, cfg.NZ = original_nx, original_ny, original_nz

def test_multiple_projections():
    """
    Test that repeated projections converge (or don't).

    If operators are consistent, one projection should be enough.
    If inconsistent, divergence will remain even after multiple projections.
    """
    print("\n" + "="*70)
    print("REPEATED PROJECTION TEST")
    print("="*70 + "\n")

    original_nx, original_ny, original_nz = cfg.NX, cfg.NY, cfg.NZ
    cfg.NX, cfg.NY, cfg.NZ = 32, 32, 20

    try:
        g = Grid()

        # Random field
        np.random.seed(42)
        u = np.random.randn(g.nx, g.ny, g.nz) * 10.0
        v = np.random.randn(g.nx, g.ny, g.nz) * 10.0
        w = np.random.randn(g.nx, g.ny, g.nz + 1) * 5.0
        w[:, :, 0] = 0.0
        w[:, :, -1] = 0.0

        div_initial = compute_divergence(u, v, w, g)
        max_div_initial = np.abs(div_initial).max()

        print(f"Initial: max|div| = {max_div_initial:.6e} s⁻¹")

        dt = 1.0
        for i in range(5):
            phi = project(u, v, w, g, dt)
            div = compute_divergence(u, v, w, g)
            max_div = np.abs(div).max()
            reduction = max_div / max_div_initial
            print(f"After projection {i+1}: max|div| = {max_div:.6e} s⁻¹ ({reduction*100:.1f}% of initial)")

        print("\nIf projections converge slowly, operators are inconsistent.")
        print("With C-grid, one projection should reach machine precision.\n")

    finally:
        cfg.NX, cfg.NY, cfg.NZ = original_nx, original_ny, original_nz

if __name__ == '__main__':
    print("\n" + "="*70)
    print("PRESSURE PROJECTION DIAGNOSTIC TESTS")
    print("="*70)
    print("\nThese tests diagnose the pressure projection problem.")
    print("They should FAIL before C-grid implementation.")
    print("They should PASS after C-grid implementation.")
    print("="*70)

    test_projection_divergence_removal()
    test_multiple_projections()

    print("\n" + "="*70)
    print("Next step: Implement C-grid staggering (Phase 1.1)")
    print("="*70 + "\n")
