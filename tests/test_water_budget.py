"""
Per-process water budget analysis for full LES model.

Measures domain-total water immediately before and after EACH process:
- Advection (RK3 dynamics)
- SGS diffusion (included in dynamics)
- Clipping (after each RK3 stage)
- Pressure projection
- Microphysics
- Sedimentation (+ surface precip removal)
- Subsidence
- Nudging
- Sponge
- Surface fluxes

Reports: contribution of each process to total water drift over 30 min.
"""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config as cfg
from grid import Grid
from state import State
from dynamics import compute_tendencies
from pressure import project
from microphysics import saturation_adjustment, warm_rain, sedimentation
from forcing import apply_subsidence, apply_profile_nudging, apply_sponge, build_flux_map
from surface_layer import apply_surface_layer
import copy

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
cfg.W_SUBS = 0.0  # No subsidence (to isolate processes)
cfg.TAU_NUDGE_T = 1e10  # No nudging
cfg.TAU_NUDGE_Q = 1e10

# Create grid and state
grid = Grid(terrain=None)
state = State(grid)

print("="*80)
print("PER-PROCESS WATER BUDGET ANALYSIS")
print("="*80)
print(f"Grid: {cfg.NX}x{cfg.NY}x{cfg.NZ}")
print(f"Test duration: 1800s (30 min)")
print(f"Surface fluxes: DISABLED (SHF=LHF=0)")
print(f"Subsidence/nudging: DISABLED (isolate processes)")
print()

# Generate surface flux maps (zero)
shf_map, lhf_map = build_flux_map(grid)

# Initialize with warm moist bubble
x_c = cfg.NX * cfg.DX / 2.0
y_c = cfg.NY * cfg.DY / 2.0
z_c = 1000.0
r_bubble = 1000.0

x = np.arange(cfg.NX) * cfg.DX + cfg.DX / 2.0
y = np.arange(cfg.NY) * cfg.DY + cfg.DY / 2.0
z = grid.z

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
R = np.sqrt((X - x_c)**2 + (Y - y_c)**2 + (Z - z_c)**2)

theta_pert = 2.0 * np.exp(-0.5 * (R / r_bubble)**2)
qv_pert = 0.002 * np.exp(-0.5 * (R / r_bubble)**2)

state.theta[:] += theta_pert
state.qv[:] += qv_pert
state.qv = np.maximum(state.qv, 0.0)
state.qc[:] = 0.0
state.qr[:] = 0.0
state._update_conserved_variables()

# Water budget computation
rho0_3d = state.rho0[np.newaxis, np.newaxis, :]
dz_3d = grid.dz_face[np.newaxis, np.newaxis, :]
dx = grid.dx
dy = grid.dy

def compute_domain_water(s):
    """Compute total water in domain (kg), NOT including surface precip."""
    qt = s.qv + s.qc + s.qr
    return (rho0_3d * qt * dz_3d * dx * dy).sum()

# Initialize budget accumulators (kg)
budget = {
    'advection': 0.0,
    'clipping_stage1': 0.0,
    'clipping_stage2': 0.0,
    'clipping_stage3': 0.0,
    'projection': 0.0,
    'microphysics': 0.0,
    'sedimentation': 0.0,
    'surface_precip': 0.0,
    'subsidence': 0.0,
    'nudging': 0.0,
    'sponge': 0.0,
    'surface_fluxes': 0.0,
    'boundary_conditions': 0.0,
}

water_initial = compute_domain_water(state)
accumulated_precip_total = 0.0

print(f"Initial domain water: {water_initial:.6f} kg")
print()
print("Running 30-min simulation with per-process budget tracking...")
print()

# Time integration
t = 0.0
dt = cfg.DT
n_steps = int(1800.0 / dt)

# Manual RK3 with budget tracking
def _copy_state_fields(s):
    return {
        'u': s.u.copy(),
        'v': s.v.copy(),
        'w': s.w.copy(),
        'theta_l': s.theta_l.copy(),
        'qt': s.qt.copy(),
        'qr': s.qr.copy(),
    }

def _update_state_fields(s, s0, tend, dt_eff):
    s.u[:] = s0['u'] + dt_eff * tend['u']
    s.v[:] = s0['v'] + dt_eff * tend['v']
    s.w[:] = s0['w'] + dt_eff * tend['w']
    s.theta_l[:] = s0['theta_l'] + dt_eff * tend['theta_l']
    s.qt[:] = s0['qt'] + dt_eff * tend['qt']
    s.qr[:] = s0['qr'] + dt_eff * tend['qr']
    s._update_diagnostic_variables()

for step in range(n_steps):
    # Save initial state
    s0 = _copy_state_fields(state)
    alpha = [1.0/3.0, 1.0/2.0, 1.0]

    # ── RK3 stages ────────────────────────────────────────────────────────────
    for stage in range(3):
        # Advection + SGS diffusion (combined in dynamics)
        w_before = compute_domain_water(state)
        tend = compute_tendencies(state, grid)
        _update_state_fields(state, s0, tend, alpha[stage] * dt)
        w_after = compute_domain_water(state)
        budget['advection'] += (w_after - w_before)

        # Clipping
        w_before = compute_domain_water(state)
        state.qc = np.maximum(state.qc, 0.0)
        state.qr = np.maximum(state.qr, 0.0)
        state.qv = np.maximum(state.qv, 0.0)
        w_after = compute_domain_water(state)
        budget[f'clipping_stage{stage+1}'] += (w_after - w_before)

        # Pressure projection (should conserve mass exactly)
        w_before = compute_domain_water(state)
        project(state.u, state.v, state.w, grid, dt, rho0=state.rho0)
        w_after = compute_domain_water(state)
        budget['projection'] += (w_after - w_before)

        # Boundary conditions after projection
        state.w[:, :, 0] = 0.0
        state.w[:, :, -1] = 0.0

    # ── Physics (applied once at end) ─────────────────────────────────────────

    # Surface layer (fluxes disabled, but still applies momentum mixing)
    w_before = compute_domain_water(state)
    ramp_factor = min(1.0, t / cfg.FLUX_RAMP_TIME) if cfg.FLUX_RAMP_TIME > 0 else 1.0
    shf_ramped = shf_map * ramp_factor
    lhf_ramped = lhf_map * ramp_factor
    apply_surface_layer(state, grid, dt, shf_ramped, lhf_ramped)
    state._update_conserved_variables()
    w_after = compute_domain_water(state)
    budget['surface_fluxes'] += (w_after - w_before)

    # Microphysics (saturation adjustment + warm rain, NO sedimentation yet)
    w_before = compute_domain_water(state)
    saturation_adjustment(state, dt)
    warm_rain(state, grid, dt)
    state._update_conserved_variables()
    w_after = compute_domain_water(state)
    budget['microphysics'] += (w_after - w_before)

    # Sedimentation (removes rain from domain)
    w_before = compute_domain_water(state)
    sedimentation(state, grid, dt)
    w_after = compute_domain_water(state)
    budget['sedimentation'] += (w_after - w_before)

    # Track surface precip separately (water that left domain)
    precip_this_step = state.surface_precip_flux.mean() * (dt / 3600.0)  # mm
    precip_water = precip_this_step * (cfg.NX * cfg.NY) * dx * dy  # kg
    budget['surface_precip'] -= precip_water  # Negative (water leaving)
    accumulated_precip_total += precip_water

    # Subsidence (disabled, but track anyway)
    w_before = compute_domain_water(state)
    apply_subsidence(state, grid, dt)
    state._update_conserved_variables()
    w_after = compute_domain_water(state)
    budget['subsidence'] += (w_after - w_before)

    # Nudging (disabled, but track anyway)
    w_before = compute_domain_water(state)
    apply_profile_nudging(state, grid, dt)
    state._update_conserved_variables()
    w_after = compute_domain_water(state)
    budget['nudging'] += (w_after - w_before)

    # Sponge layer
    w_before = compute_domain_water(state)
    apply_sponge(state, grid, dt)
    state._update_conserved_variables()
    w_after = compute_domain_water(state)
    budget['sponge'] += (w_after - w_before)

    # Final boundary conditions
    w_before = compute_domain_water(state)
    state.u[-1, :, :] = state.u[0, :, :]
    state.v[:, -1, :] = state.v[:, 0, :]
    state.w[:, :, 0] = 0.0
    state.w[:, :, -1] = 0.0
    state.qt = np.maximum(state.qt, 0.0)
    state.qr = np.maximum(state.qr, 0.0)
    state._update_diagnostic_variables()
    w_after = compute_domain_water(state)
    budget['boundary_conditions'] += (w_after - w_before)

    t += dt

# Final water
water_final = compute_domain_water(state)
total_drift = water_final - water_initial

print()
print("="*80)
print("PER-PROCESS WATER BUDGET (30 minutes)")
print("="*80)
print(f"Initial domain water:        {water_initial:18.6f} kg")
print(f"Final domain water:          {water_final:18.6f} kg")
print(f"Total drift (in domain):     {total_drift:18.6f} kg ({total_drift/water_initial*100:+.6f}%)")
print(f"Surface precip (left domain): {-accumulated_precip_total:18.6f} kg ({-accumulated_precip_total/water_initial*100:.6f}%)")
print()
print("Per-process contributions (kg):")
print("-"*80)
print(f"{'Process':<25} {'Water change (kg)':>20} {'% of initial':>15} {'% of drift':>15}")
print("-"*80)

# Sort by absolute contribution
sorted_budget = sorted(budget.items(), key=lambda x: abs(x[1]), reverse=True)

for process, delta in sorted_budget:
    pct_initial = delta / water_initial * 100.0
    pct_drift = delta / total_drift * 100.0 if abs(total_drift) > 1e-6 else 0.0
    print(f"{process:<25} {delta:+20.6e} {pct_initial:+15.6f} {pct_drift:+15.1f}")

print("-"*80)

# Sum of all processes
sum_processes = sum(budget.values())
print(f"{'Sum of all processes':<25} {sum_processes:+20.6e} {sum_processes/water_initial*100:+15.6f} {sum_processes/total_drift*100 if abs(total_drift)>1e-6 else 0:+15.1f}")
print(f"{'Measured total drift':<25} {total_drift:+20.6e} {total_drift/water_initial*100:+15.6f} {'100.0':>15}")
print(f"{'Residual (error)':<25} {total_drift - sum_processes:+20.6e} {(total_drift-sum_processes)/water_initial*100:+15.6f}")

print()
print("="*80)
print("INTERPRETATION")
print("="*80)
print("Positive values: processes that ADD water to domain")
print("Negative values: processes that REMOVE water from domain")
print()
print("Expected:")
print("  - Surface precip (sedimentation): negative (rain leaves domain)")
print("  - Clipping: positive (removes negatives, adds water)")
print("  - Advection: should be near zero (conservative scheme)")
print("  - Projection: should be exactly zero (mass-conserving)")
print("  - Microphysics: should be zero (phase changes only)")
print("="*80)
