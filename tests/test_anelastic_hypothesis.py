"""
Test anelastic consistency hypothesis.

Compares rho0-weighted vs unweighted water budget for advection.
If advection conserves sum(qt*dV) but not sum(rho0*qt*dV), then the
issue is Boussinesq/anelastic inconsistency, not a numerical leak.
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

print("="*80)
print("ANELASTIC CONSISTENCY TEST")
print("="*80)
print(f"Grid: {cfg.NX}x{cfg.NY}x{cfg.NZ}")
print(f"Test duration: 1800s (30 min)")
print()
print("Hypothesis: advection conserves unweighted sum(qt*dV) but not rho0-weighted")
print("            sum(rho0*qt*dV), indicating Boussinesq/anelastic inconsistency")
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

# Budget computation
rho0_3d = state.rho0[np.newaxis, np.newaxis, :]
dz_3d = grid.dz_face[np.newaxis, np.newaxis, :]
dx = grid.dx
dy = grid.dy
dV = dx * dy * dz_3d  # Volume element

def compute_water_weighted(s):
    """Compute rho0-weighted water: sum(rho0 * qt * dV) in kg."""
    qt = s.qv + s.qc + s.qr
    return (rho0_3d * qt * dV).sum()

def compute_water_unweighted(s):
    """Compute unweighted water: sum(qt * dV) in kg*m³."""
    qt = s.qv + s.qc + s.qr
    return (qt * dV).sum()

# Initialize budgets
budget_weighted = {
    'advection': 0.0,
    'other_processes': 0.0,
}

budget_unweighted = {
    'advection': 0.0,
    'other_processes': 0.0,
}

water_weighted_initial = compute_water_weighted(state)
water_unweighted_initial = compute_water_unweighted(state)

print(f"Initial rho0-weighted water:  {water_weighted_initial:.6f} kg")
print(f"Initial unweighted water:     {water_unweighted_initial:.6e} kg*m³")
print()
print("Running 30-min simulation...")
print()

# Time integration
t = 0.0
dt = cfg.DT
n_steps = int(1800.0 / dt)

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
    # Store state before RK3
    water_weighted_before_step = compute_water_weighted(state)
    water_unweighted_before_step = compute_water_unweighted(state)

    # Save initial state for RK3
    s0 = _copy_state_fields(state)
    alpha = [1.0/3.0, 1.0/2.0, 1.0]

    # ── RK3 stages (advection only) ───────────────────────────────────────────
    for stage in range(3):
        # Advection + SGS diffusion
        tend = compute_tendencies(state, grid)
        _update_state_fields(state, s0, tend, alpha[stage] * dt)

        # Clipping
        state.qc = np.maximum(state.qc, 0.0)
        state.qr = np.maximum(state.qr, 0.0)
        state.qv = np.maximum(state.qv, 0.0)

        # Pressure projection
        project(state.u, state.v, state.w, grid, dt, rho0=state.rho0)

        # Boundary conditions
        state.w[:, :, 0] = 0.0
        state.w[:, :, -1] = 0.0

    # Water after advection (before physics)
    water_weighted_after_advection = compute_water_weighted(state)
    water_unweighted_after_advection = compute_water_unweighted(state)

    # Accumulate advection contribution
    budget_weighted['advection'] += (water_weighted_after_advection - water_weighted_before_step)
    budget_unweighted['advection'] += (water_unweighted_after_advection - water_unweighted_before_step)

    # ── Physics ───────────────────────────────────────────────────────────────
    # (These will also contribute, but we're focusing on advection)

    ramp_factor = min(1.0, t / cfg.FLUX_RAMP_TIME) if cfg.FLUX_RAMP_TIME > 0 else 1.0
    apply_surface_layer(state, grid, dt, shf_map * ramp_factor, lhf_map * ramp_factor)
    state._update_conserved_variables()

    saturation_adjustment(state, dt)
    warm_rain(state, grid, dt)
    state._update_conserved_variables()

    sedimentation(state, grid, dt)

    apply_subsidence(state, grid, dt)
    apply_profile_nudging(state, grid, dt)
    apply_sponge(state, grid, dt)
    state._update_conserved_variables()

    # Final boundary conditions
    state.u[-1, :, :] = state.u[0, :, :]
    state.v[:, -1, :] = state.v[:, 0, :]
    state.w[:, :, 0] = 0.0
    state.w[:, :, -1] = 0.0
    state.qt = np.maximum(state.qt, 0.0)
    state.qr = np.maximum(state.qr, 0.0)
    state._update_diagnostic_variables()

    # Water after all processes
    water_weighted_after_step = compute_water_weighted(state)
    water_unweighted_after_step = compute_water_unweighted(state)

    # Other processes contribution
    budget_weighted['other_processes'] += (water_weighted_after_step - water_weighted_after_advection)
    budget_unweighted['other_processes'] += (water_unweighted_after_step - water_unweighted_after_advection)

    t += dt

# Final water
water_weighted_final = compute_water_weighted(state)
water_unweighted_final = compute_water_unweighted(state)

print()
print("="*80)
print("RESULTS: WEIGHTED (rho0) vs UNWEIGHTED BUDGETS")
print("="*80)
print()
print("RHO0-WEIGHTED BUDGET (physical, used by model):")
print("-"*80)
print(f"  Initial:          {water_weighted_initial:18.6f} kg")
print(f"  Final:            {water_weighted_final:18.6f} kg")
print(f"  Total drift:      {water_weighted_final - water_weighted_initial:18.6f} kg ({(water_weighted_final - water_weighted_initial)/water_weighted_initial*100:+.6f}%)")
print()
print(f"  Advection:        {budget_weighted['advection']:18.6f} kg ({budget_weighted['advection']/water_weighted_initial*100:+.6f}%)")
print(f"  Other processes:  {budget_weighted['other_processes']:18.6f} kg ({budget_weighted['other_processes']/water_weighted_initial*100:+.6f}%)")
print()
print("UNWEIGHTED BUDGET (sum qt*dV, no rho0):")
print("-"*80)
print(f"  Initial:          {water_unweighted_initial:18.6e} kg*m³")
print(f"  Final:            {water_unweighted_final:18.6e} kg*m³")
print(f"  Total drift:      {water_unweighted_final - water_unweighted_initial:18.6e} kg*m³ ({(water_unweighted_final - water_unweighted_initial)/water_unweighted_initial*100:+.6f}%)")
print()
print(f"  Advection:        {budget_unweighted['advection']:18.6e} kg*m³ ({budget_unweighted['advection']/water_unweighted_initial*100:+.6f}%)")
print(f"  Other processes:  {budget_unweighted['other_processes']:18.6e} kg*m³ ({budget_unweighted['other_processes']/water_unweighted_initial*100:+.6f}%)")
print()
print("="*80)
print("HYPOTHESIS TEST")
print("="*80)

advection_weighted_pct = abs(budget_weighted['advection'] / water_weighted_initial * 100.0)
advection_unweighted_pct = abs(budget_unweighted['advection'] / water_unweighted_initial * 100.0)

print(f"Advection contribution (rho0-weighted):  {advection_weighted_pct:.6f}%")
print(f"Advection contribution (unweighted):     {advection_unweighted_pct:.6f}%")
print()

if advection_unweighted_pct < 0.01 and advection_weighted_pct > 0.1:
    print("✓ HYPOTHESIS CONFIRMED:")
    print("  - Unweighted advection is ~0 (numerically conservative)")
    print("  - Weighted advection has significant drift (anelastic inconsistency)")
    print()
    print("ROOT CAUSE: Model uses div(u)=0 but should use div(rho0*u)=0")
    print()
    print("FIX REQUIRED:")
    print("  1. Pressure projection: enforce div(rho0*u) = 0")
    print("  2. Scalar advection: use anelastic form d(phi)/dt = -(1/rho0)*div(rho0*u*phi)")
    print("  3. Momentum advection: use rho0-weighted vertical fluxes")
elif advection_unweighted_pct > 0.1:
    print("✗ HYPOTHESIS REJECTED:")
    print("  - Both weighted and unweighted advection have drift")
    print("  - This indicates a true numerical leak in the advection scheme")
else:
    print("? INCONCLUSIVE:")
    print("  - Both budgets show small errors")
    print("  - Need more diagnostics to identify source")

print("="*80)
