"""
Test water conservation in the FULL MODEL (dynamics + microphysics).

This is the real test: dynamics/advection creates small negative values
at sharp cloud edges, and clipping in rk3_step can leak water.

Requirements:
- Dynamics/advection ON
- Warm bubble to trigger cloud formation
- Sedimentation ON (rain can leave domain)
- Surface fluxes OFF (no water input)
- 30 minutes simulation
- Track: domain-total water + accumulated surface precip
- Report: drift, and water added by clipping per step
- Acceptable: < 0.1% over 30 min
"""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config as cfg
from grid import Grid
from state import State
from timestepping import rk3_step
from forcing import build_flux_map

# Override config for test
cfg.NX = 32
cfg.NY = 32
cfg.NZ = 80
cfg.DX = 100.0
cfg.DY = 100.0
cfg.Z_TOP = 5000.0
cfg.DT = 2.0
cfg.SHF = 0.0  # No surface fluxes
cfg.LHF = 0.0
cfg.W_SUBS = 0.0  # No subsidence
cfg.TAU_NUDGE_T = 1e10  # No nudging
cfg.TAU_NUDGE_Q = 1e10

# Create grid and state
grid = Grid(terrain=None)
state = State(grid)

print("="*70)
print("FULL MODEL WATER CONSERVATION TEST")
print("="*70)
print(f"Grid: {cfg.NX}x{cfg.NY}x{cfg.NZ}")
print(f"Test duration: 1800s (30 min)")
print(f"Surface fluxes: DISABLED (SHF=LHF=0)")
print(f"Sedimentation: ENABLED (rain can leave domain)")
print(f"Dynamics/advection: ENABLED")
print(f"Initial condition: Warm moist bubble to trigger cloud")
print()

# Generate surface flux maps (zero)
shf_map, lhf_map = build_flux_map(grid)

# Initialize with warm moist bubble to trigger cloud formation
# Bubble centered at domain center, z = 1000m
x_c = cfg.NX * cfg.DX / 2.0
y_c = cfg.NY * cfg.DY / 2.0
z_c = 1000.0
r_bubble = 1000.0  # m, bubble radius

# Create coordinate arrays
x = np.arange(cfg.NX) * cfg.DX + cfg.DX / 2.0
y = np.arange(cfg.NY) * cfg.DY + cfg.DY / 2.0
z = grid.z

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Distance from bubble center
R = np.sqrt((X - x_c)**2 + (Y - y_c)**2 + (Z - z_c)**2)

# Bubble perturbations (smooth Gaussian)
theta_pert = 2.0 * np.exp(-0.5 * (R / r_bubble)**2)  # K
qv_pert = 0.002 * np.exp(-0.5 * (R / r_bubble)**2)   # kg/kg (2 g/kg)

# Apply perturbations
state.theta[:] += theta_pert
state.qv[:] += qv_pert

# Ensure non-negative
state.qv = np.maximum(state.qv, 0.0)
state.qc[:] = 0.0
state.qr[:] = 0.0

# Update conserved variables
state._update_conserved_variables()

# Compute initial total water
# Total water = ∫∫∫ rho0 * (qv + qc + qr) * dx * dy * dz + accumulated surface precip
rho0_3d = state.rho0[np.newaxis, np.newaxis, :]
dz_3d = grid.dz_face[np.newaxis, np.newaxis, :]
dx = grid.dx
dy = grid.dy

def compute_domain_water(s, precip_accum):
    """Compute total water in domain + accumulated precip (kg)."""
    qt = s.qv + s.qc + s.qr
    domain_water = (rho0_3d * qt * dz_3d * dx * dy).sum()
    # Surface precip: mm * (1 kg/m² / mm) * area
    precip_water = precip_accum.sum() * dx * dy  # kg
    return domain_water + precip_water

accumulated_precip = np.zeros((cfg.NX, cfg.NY))  # mm
total_water_initial = compute_domain_water(state, accumulated_precip)

print(f"Initial conditions:")
print(f"  Domain-mean qv: {state.qv.mean()*1e3:.3f} g/kg")
print(f"  Domain-mean qc: {state.qc.mean()*1e3:.3f} g/kg")
print(f"  Domain-mean qr: {state.qr.mean()*1e3:.3f} g/kg")
print(f"  Total water (domain + precip): {total_water_initial:.6f} kg")
print()

# Time integration
t = 0.0
dt = cfg.DT
n_steps = int(1800.0 / dt)  # 30 minutes

# Track clipping losses
total_clipping_loss = 0.0

print("Time evolution:")
print(f"{'t(min)':>7} {'qv(g/kg)':>10} {'qc(g/kg)':>10} {'qr(g/kg)':>10} {'total_water(kg)':>18} {'error(%)':>10} {'clip_loss(kg)':>14}")
print("-"*100)

for step in range(n_steps):
    # Store pre-step water to measure clipping loss
    qt_before = state.qv + state.qc + state.qr
    water_before_clip = (rho0_3d * qt_before * dz_3d * dx * dy).sum()

    # RK3 step (full model with dynamics and microphysics)
    rk3_step(state, grid, dt, shf_map, lhf_map, t=t)

    # Measure water after step (before accumulating precip)
    qt_after = state.qv + state.qc + state.qr
    water_after_step = (rho0_3d * qt_after * dz_3d * dx * dy).sum()

    # Clipping loss = water before - water after - precip leaving
    # (surface precip is already removed from domain during sedimentation)
    # So clipping loss is just the difference not accounted for by precip
    precip_this_step = state.surface_precip_flux * (dt / 3600.0)  # mm
    precip_water_this_step = precip_this_step.sum() * dx * dy  # kg

    # Water should decrease by precip amount if no leaks
    expected_water = water_before_clip - precip_water_this_step
    clipping_loss_this_step = expected_water - water_after_step
    total_clipping_loss += clipping_loss_this_step

    # Accumulate precipitation
    accumulated_precip += precip_this_step

    t += dt

    # Compute diagnostics every 2 minutes
    if step % 60 == 0 or step == n_steps - 1:
        qv_mean = state.qv.mean() * 1e3
        qc_mean = state.qc.mean() * 1e3
        qr_mean = state.qr.mean() * 1e3

        total_water_current = compute_domain_water(state, accumulated_precip)

        # Conservation error
        error_pct = abs(total_water_current - total_water_initial) / total_water_initial * 100.0

        print(f"{t/60:7.1f} {qv_mean:10.3f} {qc_mean:10.3f} {qr_mean:10.3f} {total_water_current:18.6f} {error_pct:10.6f} {total_clipping_loss:14.6e}")

# Final check
total_water_final = compute_domain_water(state, accumulated_precip)
error_pct = abs(total_water_final - total_water_initial) / total_water_initial * 100.0

print()
print("="*70)
print("MASS BALANCE CHECK")
print("="*70)
print(f"Initial total water (domain + precip): {total_water_initial:.6f} kg")
print(f"Final total water (domain + precip):   {total_water_final:.6f} kg")
print(f"Absolute error:                        {abs(total_water_final - total_water_initial):.6e} kg")
print(f"Relative error:                        {error_pct:.6f}%")
print()
print(f"Total water lost to clipping:          {total_clipping_loss:.6e} kg")
print(f"Clipping loss as % of initial:         {abs(total_clipping_loss / total_water_initial * 100.0):.6f}%")
print(f"Average clipping loss per step:        {total_clipping_loss / n_steps:.6e} kg/step")
print()
print(f"Total precipitation:                   {accumulated_precip.sum() / (cfg.NX * cfg.NY):.3f} mm (domain mean)")
print()

if error_pct < 0.1:
    print(f"✓ PASS: Water conserved within 0.1% (error = {error_pct:.6f}%)")
else:
    print(f"✗ FAIL: Water conservation error > 0.1% (error = {error_pct:.6f}%)")
    print()
    if abs(total_clipping_loss / total_water_initial * 100.0) > 0.05:
        print("⚠  WARNING: Clipping loss is significant (> 0.05%)")
        print("   Consider implementing conservative clipping:")
        print("   - Subtract clipped water from neighboring cells")
        print("   - Or use flux-limiter advection schemes")

print("="*70)
