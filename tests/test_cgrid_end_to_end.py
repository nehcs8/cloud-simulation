"""
End-to-end test for C-grid implementation.

Tests that the simulation runs for a few timesteps without crashing
and that divergence remains small throughout.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grid import Grid
from state import State
from dynamics import compute_tendencies
from pressure import project
from microphysics import apply_microphysics
import config as cfg

def compute_divergence_cgrid(u, v, w, g: Grid) -> np.ndarray:
    """Compute divergence on C-grid."""
    dx, dy = g.dx, g.dy
    dz_f = g.dz_face

    div_u = (u[1:, :, :] - u[:-1, :, :]) / dx
    div_v = (v[:, 1:, :] - v[:, :-1, :]) / dy
    div_w = (w[:, :, 1:] - w[:, :, :-1]) / dz_f[np.newaxis, np.newaxis, :]

    return div_u + div_v + div_w


def test_cgrid_timestep():
    """Test that C-grid simulation runs for several timesteps."""
    print("=" * 70)
    print("C-GRID END-TO-END TEST")
    print("=" * 70)

    # Create grid and state
    g = Grid(terrain=None)
    s = State(g, restart_file=None)

    print(f"\nGrid: {g.nx} × {g.ny} × {g.nz}")
    print(f"Array shapes:")
    print(f"  u: {s.u.shape} (on x-faces)")
    print(f"  v: {s.v.shape} (on y-faces)")
    print(f"  w: {s.w.shape} (on z-faces)")
    print(f"  theta: {s.theta.shape} (cell-centered)")

    # Run a few timesteps
    dt = 1.0
    n_steps = 5

    print(f"\nRunning {n_steps} timesteps (dt={dt}s)...")

    for step in range(n_steps):
        # Compute tendencies
        tend = compute_tendencies(s, g)

        # Forward Euler update
        s.u += dt * tend["u"]
        s.v += dt * tend["v"]
        s.w += dt * tend["w"]
        s.theta += dt * tend["theta"]
        s.qv += dt * tend["qv"]
        s.qc += dt * tend["qc"]
        s.qr += dt * tend["qr"]

        # Clip hydrometeors
        s.qc = np.maximum(s.qc, 0.0)
        s.qr = np.maximum(s.qr, 0.0)

        # Microphysics
        apply_microphysics(s, g, dt)

        # Pressure projection
        project(s.u, s.v, s.w, g, dt, rho0=s.rho0)

        # Check divergence
        div = compute_divergence_cgrid(s.u, s.v, s.w, g)
        div_max = np.abs(div).max()

        # Get max velocities for reporting
        u_max = np.abs(s.u).max()
        v_max = np.abs(s.v).max()
        w_max = np.abs(s.w).max()

        print(f"  Step {step+1}: |u|_max={u_max:.3f} m/s, |v|_max={v_max:.3f} m/s, "
              f"|w|_max={w_max:.3f} m/s, max|∇·u|={div_max:.2e}")

        # Check that divergence is small
        if div_max > 1e-10:
            print(f"\n✗ FAIL: Divergence too large at step {step+1}: {div_max:.2e}")
            return False

    print(f"\n{'='*70}")
    print("✓ PASS: C-grid simulation completed successfully!")
    print(f"  All {n_steps} timesteps completed without error")
    print(f"  Divergence remained below 1e-10 throughout")
    print("  C-grid implementation is working end-to-end")

    return True


if __name__ == "__main__":
    success = test_cgrid_timestep()
    sys.exit(0 if success else 1)
