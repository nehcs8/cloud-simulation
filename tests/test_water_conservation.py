"""
Test water conservation in closed box.

Requirement: With no surface fluxes, total water (qv+qc+qr) must be conserved
to < 0.1% over 30 minutes.

This tests:
1. Saturation adjustment conserves water (qv ↔ qc)
2. Warm rain processes conserve water (qc → qr)
3. Sedimentation is disabled (closed box, no rain exit)
"""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config as cfg
from grid import Grid
from state import State
from microphysics import saturation_adjustment, warm_rain

# Override config for test
cfg.NX = 16
cfg.NY = 16
cfg.NZ = 60
cfg.DX = 100.0
cfg.DY = 100.0
cfg.Z_TOP = 5000.0
cfg.DT = 2.0
cfg.SHF = 0.0  # No surface fluxes
cfg.LHF = 0.0

# Create grid and state
grid = Grid(terrain=None)
state = State(grid)

print("="*70)
print("WATER CONSERVATION TEST (Closed Box)")
print("="*70)
print(f"Grid: {cfg.NX}x{cfg.NY}x{cfg.NZ}")
print(f"Test duration: 1800s (30 min)")
print(f"Surface fluxes: DISABLED (SHF=LHF=0)")
print(f"Sedimentation: DISABLED (closed box)")
print()

# Initialize with some moisture and cloud water
# Add random perturbations to trigger microphysical processes
np.random.seed(42)
state.qv[:] = 0.012 + 0.002 * np.random.randn(cfg.NX, cfg.NY, cfg.NZ)  # ~12 g/kg
state.qc[:] = 0.0005 * np.maximum(0, np.random.randn(cfg.NX, cfg.NY, cfg.NZ))  # 0-1 g/kg
state.qr[:] = 0.0001 * np.maximum(0, np.random.randn(cfg.NX, cfg.NY, cfg.NZ))  # 0-0.1 g/kg

# Ensure non-negative
state.qv = np.maximum(state.qv, 0.0)
state.qc = np.maximum(state.qc, 0.0)
state.qr = np.maximum(state.qr, 0.0)

# Set theta to something realistic
state.theta[:] = 1.0 + 0.5 * np.random.randn(cfg.NX, cfg.NY, cfg.NZ)

# Compute initial total water
# Total water per unit volume: rho0 * (qv + qc + qr)
# Integrate over domain: ∫∫∫ rho0 * qt * dx * dy * dz
rho0_3d = state.rho0[np.newaxis, np.newaxis, :]
dz_3d = grid.dz_face[np.newaxis, np.newaxis, :]
dx = grid.dx
dy = grid.dy

qt_initial = state.qv + state.qc + state.qr
total_water_initial = (rho0_3d * qt_initial * dz_3d * dx * dy).sum()  # kg

print(f"Initial conditions:")
print(f"  Domain-mean qv: {state.qv.mean()*1e3:.3f} g/kg")
print(f"  Domain-mean qc: {state.qc.mean()*1e3:.3f} g/kg")
print(f"  Domain-mean qr: {state.qr.mean()*1e3:.3f} g/kg")
print(f"  Total water in domain: {total_water_initial:.6f} kg")
print()

# Time integration
t = 0.0
dt = cfg.DT
n_steps = int(1800.0 / dt)  # 30 minutes

print("Time evolution:")
print(f"{'t(min)':>7} {'qv(g/kg)':>10} {'qc(g/kg)':>10} {'qr(g/kg)':>10} {'total_water(kg)':>18} {'error(%)':>10}")
print("-"*70)

for step in range(n_steps):
    # Apply microphysics (NO sedimentation - closed box!)
    # Only saturation adjustment and warm rain
    saturation_adjustment(state, dt)
    warm_rain(state, grid, dt)

    # Enforce non-negative
    state.qv = np.maximum(state.qv, 0.0)
    state.qc = np.maximum(state.qc, 0.0)
    state.qr = np.maximum(state.qr, 0.0)

    t += dt

    # Compute diagnostics every 2 minutes
    if step % 60 == 0 or step == n_steps - 1:
        qv_mean = state.qv.mean() * 1e3
        qc_mean = state.qc.mean() * 1e3
        qr_mean = state.qr.mean() * 1e3

        qt_current = state.qv + state.qc + state.qr
        total_water_current = (rho0_3d * qt_current * dz_3d * dx * dy).sum()

        # Conservation error
        error_pct = abs(total_water_current - total_water_initial) / total_water_initial * 100.0

        print(f"{t/60:7.1f} {qv_mean:10.3f} {qc_mean:10.3f} {qr_mean:10.3f} {total_water_current:18.6f} {error_pct:10.6f}")

# Final check
qt_final = state.qv + state.qc + state.qr
total_water_final = (rho0_3d * qt_final * dz_3d * dx * dy).sum()
error_pct = abs(total_water_final - total_water_initial) / total_water_initial * 100.0

print()
print("="*70)
print("MASS BALANCE CHECK")
print("="*70)
print(f"Initial total water: {total_water_initial:.6f} kg")
print(f"Final total water:   {total_water_final:.6f} kg")
print(f"Absolute error:      {abs(total_water_final - total_water_initial):.6e} kg")
print(f"Relative error:      {error_pct:.6f}%")
print()

if error_pct < 0.1:
    print(f"✓ PASS: Water conserved within 0.1% (error = {error_pct:.6f}%)")
else:
    print(f"✗ FAIL: Water conservation error > 0.1% (error = {error_pct:.6f}%)")

print("="*70)
