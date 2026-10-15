"""
Test rain sedimentation fix.

Requirement: A column with qr=1 g/kg at 2 km and no wind must:
1. Reach the ground in a few minutes
2. Column-integrated water lost must equal accumulated surface precip within 1%
"""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config as cfg
from grid import Grid
from state import State
from microphysics import sedimentation

# Override config for test
cfg.NX = 8
cfg.NY = 8
cfg.NZ = 80
cfg.DX = 100.0
cfg.DY = 100.0
cfg.Z_TOP = 5000.0
cfg.DT = 2.0

# Create grid and state
grid = Grid(terrain=None)
state = State(grid)

# Set up test: qr = 1 g/kg at z ~ 2 km, no wind
z_target = 2000.0  # m
k_target = np.argmin(np.abs(grid.z - z_target))
z_actual = grid.z[k_target]

print("="*70)
print("RAIN SEDIMENTATION TEST")
print("="*70)
print(f"Grid: {cfg.NX}x{cfg.NY}x{cfg.NZ}, dz = {grid.dz_face.min():.1f}-{grid.dz_face.max():.1f}m")
print(f"Initial condition: qr = 1.0 g/kg at k={k_target}, z={z_actual:.1f}m")
print(f"No wind (u=v=w=0)")
print()

# Initialize: qr=1 g/kg at target level, zero everywhere else
state.qr[:, :, :] = 0.0
state.qr[:, :, k_target] = 0.001  # 1 g/kg

# Zero out all velocity
state.u[:] = 0.0
state.v[:] = 0.0
state.w[:] = 0.0

# Compute initial column-integrated rain water (kg/m²)
# Integrate: ∫ rho0 * qr * dz
rho0_3d = state.rho0[np.newaxis, np.newaxis, :]
dz_3d = grid.dz_face[np.newaxis, np.newaxis, :]
rain_water_initial = (rho0_3d * state.qr * dz_3d).sum(axis=2)  # (nx, ny) kg/m²
total_water_initial = rain_water_initial.mean()

print(f"Initial column rain water: {total_water_initial:.6f} kg/m²")
print()

# Time integration: run until rain reaches ground or t > 600s
t = 0.0
dt = cfg.DT
accumulated_precip = np.zeros((cfg.NX, cfg.NY))  # mm

print("Time evolution:")
print(f"{'t(s)':>6} {'qr_max(g/kg)':>12} {'z_max(m)':>10} {'col_water(kg/m²)':>16} {'precip(mm)':>12}")
print("-"*70)

for step in range(500):  # max 1000 seconds
    # Store precip flux before sedimentation
    precip_before = state.surface_precip_flux.copy()

    # Apply sedimentation
    sedimentation(state, grid, dt)

    # Accumulate precipitation (mm/hr * dt/3600 = mm)
    accumulated_precip += state.surface_precip_flux * (dt / 3600.0)

    t += dt

    # Compute diagnostics every 10 steps
    if step % 10 == 0:
        qr_max = state.qr.max() * 1e3  # g/kg
        # Find altitude of max qr
        if qr_max > 1e-6:
            k_max = np.unravel_index(np.argmax(state.qr), state.qr.shape)[2]
            z_max = grid.z[k_max]
        else:
            z_max = 0.0

        # Column-integrated rain water
        rain_water = (rho0_3d * state.qr * dz_3d).sum(axis=2).mean()

        # Mean accumulated precip
        mean_precip = accumulated_precip.mean()

        print(f"{t:6.0f} {qr_max:12.6f} {z_max:10.1f} {rain_water:16.6f} {mean_precip:12.6f}")

        # Stop if qr_max drops below threshold
        if qr_max < 1e-4:  # 0.0001 g/kg
            print()
            print(f"Rain has reached ground at t={t:.1f}s")
            break

# Final mass balance check
rain_water_final = (rho0_3d * state.qr * dz_3d).sum(axis=2).mean()
total_precip = accumulated_precip.mean()  # mm

# Convert mm to kg/m²: 1 mm = 1 kg/m²
water_lost = total_water_initial - rain_water_final
precip_kg_m2 = total_precip * 1.0  # mm * (1 kg/m² / 1 mm)

print()
print("="*70)
print("MASS BALANCE CHECK")
print("="*70)
print(f"Initial column water:     {total_water_initial:.6f} kg/m²")
print(f"Final column water:       {rain_water_final:.6f} kg/m²")
print(f"Water lost from column:   {water_lost:.6f} kg/m²")
print(f"Accumulated precip:       {precip_kg_m2:.6f} kg/m² ({total_precip:.6f} mm)")
print()

# Mass conservation error
if water_lost > 1e-10:
    error_pct = abs(precip_kg_m2 - water_lost) / water_lost * 100.0
    print(f"Mass conservation error:  {error_pct:.3f}%")

    if error_pct < 1.0:
        print("✓ PASS: Mass conserved within 1%")
    else:
        print("✗ FAIL: Mass conservation error > 1%")
else:
    print("Warning: Very little water in column")

print()

# Check that rain reached ground in reasonable time
if t < 600.0 and state.qr.max() < 1e-6:
    print(f"✓ PASS: Rain reached ground in {t:.1f}s (< 600s)")
else:
    print(f"✗ FAIL: Rain did not reach ground in reasonable time")

print("="*70)
