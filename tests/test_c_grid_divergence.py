"""
Test that C-grid pressure projection removes divergence to machine precision.

This is the CRITICAL test from the rebuild plan (Phase 1.1, step 6).
Success criterion: max|∇·u| < 1e-12 after projection.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grid import Grid
from pressure import project
import config as cfg

def compute_divergence_cgrid(u, v, w, g: Grid) -> np.ndarray:
    """
    Compute divergence on C-grid.
    u: (nx+1, ny, nz) on x-faces
    v: (nx, ny+1, nz) on y-faces
    w: (nx, ny, nz+1) on z-faces
    Returns: (nx, ny, nz) cell-centered divergence
    """
    dx, dy = g.dx, g.dy
    dz_f = g.dz_face

    # du/dx: simple difference (matches pressure.py)
    div_u = (u[1:, :, :] - u[:-1, :, :]) / dx         # (nx, ny, nz)

    # dv/dy: simple difference
    div_v = (v[:, 1:, :] - v[:, :-1, :]) / dy         # (nx, ny, nz)

    # dw/dz: simple difference
    div_w = (w[:, :, 1:] - w[:, :, :-1]) / dz_f[np.newaxis, np.newaxis, :]  # (nx, ny, nz)

    return div_u + div_v + div_w


def test_divergence_free_projection():
    """Test that projection removes divergence to machine precision."""
    print("=" * 70)
    print("C-GRID DIVERGENCE REMOVAL TEST")
    print("=" * 70)

    # Create a small test grid
    g = Grid(terrain=None)
    nx, ny, nz = g.nx, g.ny, g.nz

    # Create random velocity field (C-grid staggered)
    print(f"\nGrid: {nx} × {ny} × {nz}")
    print(f"dx={g.dx}m, dz range: {g.dz_face.min():.1f}-{g.dz_face.max():.1f}m")

    np.random.seed(42)
    u = np.random.randn(nx + 1, ny, nz) * 5.0  # on x-faces
    v = np.random.randn(nx, ny + 1, nz) * 5.0  # on y-faces
    w = np.random.randn(nx, ny, nz + 1) * 2.0  # on z-faces
    w[:, :, 0] = 0.0  # rigid lid bottom
    w[:, :, -1] = 0.0  # rigid lid top

    # Compute initial divergence
    div_before = compute_divergence_cgrid(u, v, w, g)
    div_max_before = np.abs(div_before).max()
    div_mean_before = np.abs(div_before).mean()

    print(f"\nBefore projection:")
    print(f"  max|∇·u| = {div_max_before:.6e} s⁻¹")
    print(f"  mean|∇·u| = {div_mean_before:.6e} s⁻¹")

    # Apply pressure projection
    dt = 1.0  # doesn't matter for this test
    phi = project(u, v, w, g, dt)

    # Compute final divergence
    div_after = compute_divergence_cgrid(u, v, w, g)
    div_max_after = np.abs(div_after).max()
    div_mean_after = np.abs(div_after).mean()

    print(f"\nAfter projection:")
    print(f"  max|∇·u| = {div_max_after:.6e} s⁻¹")
    print(f"  mean|∇·u| = {div_mean_after:.6e} s⁻¹")
    print(f"  reduction factor: {div_max_before / max(div_max_after, 1e-16):.2e}")

    # Success criterion from plan
    SUCCESS_THRESHOLD = 1e-12

    print(f"\n{'='*70}")
    if div_max_after < SUCCESS_THRESHOLD:
        print(f"✓ PASS: max|∇·u| < {SUCCESS_THRESHOLD:.0e}")
        print("  C-grid pressure projection is working correctly!")
        print("  Divergence and gradient stencils are properly matched.")
        return True
    else:
        print(f"✗ FAIL: max|∇·u| = {div_max_after:.2e} >= {SUCCESS_THRESHOLD:.0e}")
        print("  Problem: divergence and gradient stencils don't match.")
        print("  This is the CRITICAL bug identified in the rebuild plan.")
        return False


if __name__ == "__main__":
    success = test_divergence_free_projection()
    sys.exit(0 if success else 1)
