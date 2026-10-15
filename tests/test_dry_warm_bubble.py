"""
Dry warm bubble sanity test for anelastic consistency.

Tests that total kinetic energy is conserved (doesn't grow) after the bubble
stops rising and oscillates. This verifies energy conservation in the
anelastic formulation.

Setup:
- Initial condition: theta' = +1 K * cos²(pi*r/2R) for r < R=500m, centered at z=500m
- No moisture perturbation (qv'  = qc = qr = 0)
- No surface fluxes (SHF=LHF=0)
- No Coriolis force
- Zero initial velocity (u = v = w = 0)
- Duration: 20 minutes
- Report: KE(t), w_max, bubble-top height every 1 min

Expected: w_max peaks at 3–5 m/s, bubble stops near top of mixed layer,
          then KE slowly decays (no growth after oscillation starts).
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
cfg.ENABLE_CORIOLIS = False  # No Coriolis

# Create grid and state
grid = Grid(terrain=None)
state = State(grid)

print("="*80)
print("DRY WARM BUBBLE SANITY TEST (KE CONSERVATION)")
print("="*80)
print(f"Grid: {cfg.NX}x{cfg.NY}x{cfg.NZ}")
print(f"Test duration: 1200s (20 min)")
print(f"Surface fluxes: DISABLED (SHF=LHF=0)")
print(f"Coriolis: DISABLED")
print(f"Initial condition: theta' = +1 K * cos²(pi*r/2R), R=500m, z_c=500m")
print()

# Generate surface flux maps (zero)
shf_map, lhf_map = build_flux_map(grid)

# Initialize with dry warm bubble: theta' = +1 K * cos²(pi*r/2R) for r < R
x_c = cfg.NX * cfg.DX / 2.0
y_c = cfg.NY * cfg.DY / 2.0
z_c = 500.0
R = 500.0

x = np.arange(cfg.NX) * cfg.DX + cfg.DX / 2.0
y = np.arange(cfg.NY) * cfg.DY + cfg.DY / 2.0
z = grid.z

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
r = np.sqrt((X - x_c)**2 + (Y - y_c)**2 + (Z - z_c)**2)

# Cosine-squared bubble
theta_pert = np.where(r < R,
                      1.0 * np.cos(0.5 * np.pi * r / R)**2,
                      0.0)

# Set theta' (perturbation from theta0)
state.theta[:] = theta_pert

# DRY bubble: zero moisture everywhere (background qv0 included).
# The default sounding is supersaturated at ~950-1450 m, so keeping qv0
# would make this a moist (latent-heat-driven) bubble.
state.qv0[:] = 0.0
state.qv[:] = 0.0
state.qc[:] = 0.0
state.qr[:] = 0.0

# Zero initial velocity
state.u[:] = 0.0
state.v[:] = 0.0
state.w[:] = 0.0

# Update conserved variables
state._update_conserved_variables()

print(f"Initial conditions:")
print(f"  Base-state theta0 at surface: {state.theta0[0]:.2f} K")
print(f"  Base-state theta0 at z=500m:  {state.theta0[np.argmin(np.abs(z - 500))]:2f} K")
print(f"  Base-state theta0 at z=1km:   {state.theta0[np.argmin(np.abs(z - 1000))]:2f} K")
print(f"  Base-state theta0 at top:     {state.theta0[-1]:.2f} K")
print(f"  Max theta' (perturbation):    {state.theta.max():.3f} K")
print(f"  Mean theta' (perturbation):   {state.theta.mean():.6f} K")
print(f"  qv' = 0 (dry bubble, background qv0 only)")
print()

# Compute total kinetic energy
rho0_3d = state.rho0[np.newaxis, np.newaxis, :]
dz_3d = grid.dz_face[np.newaxis, np.newaxis, :]
dx = grid.dx
dy = grid.dy
dV = dx * dy * dz_3d  # Volume element

def compute_total_KE(s):
    """Compute total kinetic energy: 0.5 * sum(rho0 * (u^2 + v^2 + w^2) * dV)."""
    # u, v, w are on faces - need to interpolate to cell centers for KE
    u_cen = s.u_at_cell()  # (nx, ny, nz)
    v_cen = s.v_at_cell()  # (nx, ny, nz)
    w_cen = s.w_at_cell()  # (nx, ny, nz)

    KE = 0.5 * rho0_3d * (u_cen**2 + v_cen**2 + w_cen**2)
    return (KE * dV).sum()

def find_bubble_top(s):
    """Find bubble top height using w=0.1 m/s threshold."""
    w_cen = s.w_at_cell()
    w_max_horiz = w_cen.max(axis=(0, 1))  # (nz,) max w at each level
    # Find highest level where w > 0.1 m/s
    levels_active = np.where(w_max_horiz > 0.1)[0]
    if len(levels_active) == 0:
        return 0.0
    return z[levels_active[-1]]

KE_initial = compute_total_KE(state)

print(f"Initial KE: {KE_initial:.6e} J (should be 0.0)")
print()

# Time integration
t = 0.0
dt = cfg.DT
n_steps = int(1200.0 / dt)  # 20 minutes

# Track KE evolution
KE_history = []
t_history = []
w_max_history = []
bubble_top_history = []

print("Time evolution (reporting every 1 min):")
print(f"{'t(min)':>7} {'KE (J)':>18} {'dKE/dt (W)':>15} {'w_max (m/s)':>15} {'Z_top (m)':>12}")
print("-"*80)

# Report initial state
KE_current = compute_total_KE(state)
w_max = np.abs(state.w).max()
z_top = find_bubble_top(state)
KE_history.append(KE_current)
t_history.append(t)
w_max_history.append(w_max)
bubble_top_history.append(z_top)
print(f"{t/60:7.1f} {KE_current:18.6e} {0.0:+15.6e} {w_max:15.3f} {z_top:12.1f}")

# Store w_max at t=8 min for comparison
w_max_at_8min = 0.0

for step in range(1, n_steps + 1):
    # Time step
    # Dynamics only: skip surface layer, microphysics, forcing, sponge
    rk3_step(state, grid, dt, shf_map, lhf_map,
             physics_callback=lambda s, g, dt: None, t=t)
    t += dt

    # Compute diagnostics every 1 minute (30 steps)
    if step % 30 == 0:
        KE_current = compute_total_KE(state)
        w_max = np.abs(state.w).max()
        z_top = find_bubble_top(state)

        # Compute dKE/dt from last minute
        if len(KE_history) > 0:
            dKE_dt = (KE_current - KE_history[-1]) / 60.0  # W
        else:
            dKE_dt = 0.0

        KE_history.append(KE_current)
        t_history.append(t)
        w_max_history.append(w_max)
        bubble_top_history.append(z_top)

        print(f"{t/60:7.1f} {KE_current:18.6e} {dKE_dt:+15.6e} {w_max:15.3f} {z_top:12.1f}")

        # Store w_max at t=8 min for comparison
        if abs(t - 480.0) < dt:
            w_max_at_8min = w_max

# Final analysis
KE_final = KE_history[-1]
KE_max = max(KE_history)
w_max_overall = max(w_max_history)

# Find index of peak KE
idx_peak = KE_history.index(KE_max)
t_peak = t_history[idx_peak]

print()
print("="*80)
print("KE CONSERVATION ANALYSIS")
print("="*80)
print(f"Initial KE:                 {KE_initial:.6e} J")
print(f"Peak KE:                    {KE_max:.6e} J  at t={t_peak/60:.1f} min")
print(f"Final KE (t=20 min):        {KE_final:.6e} J  ({(KE_final - KE_max)/KE_max*100:+.3f}% from peak)")
print()
print(f"Peak w_max (any time):      {w_max_overall:.3f} m/s")
print(f"w_max at t=8 min:           {w_max_at_8min:.3f} m/s")
print(f"Peak bubble-top height:     {max(bubble_top_history):.1f} m")
print()

# Check if KE grows after peak (bubble stops rising and oscillates)
# Allow 2-minute window after peak to identify oscillation start
idx_check_start = min(idx_peak + 2, len(KE_history) - 1)
if idx_check_start < len(KE_history) - 1:
    # Largest rebound above the running minimum (catches late regrowth)
    KE_after_peak = np.array(KE_history[idx_check_start:])
    rebound = KE_after_peak - np.minimum.accumulate(KE_after_peak)
    KE_growth_after_peak = rebound.max() / KE_max * 100.0
else:
    KE_growth_after_peak = 0.0

print("="*80)
print("SANITY CHECK")
print("="*80)
print(f"KE growth after peak+2min:  {KE_growth_after_peak:+.6f}% of peak KE")
print()

if abs(KE_growth_after_peak) < 2.0:
    print("✓ PASS: KE does not grow significantly after bubble stops rising")
    print("        (anelastic energy conservation is working)")
else:
    print("✗ FAIL: KE grows after bubble stops rising")
    print("        (indicates energy conservation issue in anelastic formulation)")

print()
print("Expected results:")
print("  - w_max peaks at 3–5 m/s (parcel theory upper bound ~8 m/s)")
print("  - Bubble stops near top of mixed layer (~800-1200 m)")
print("  - KE decays slowly after peak (dissipation from SGS + numerical)")
print()

# Check w_max is physically reasonable
if 3.0 <= w_max_overall <= 8.0:
    print(f"✓ w_max = {w_max_overall:.2f} m/s is physically reasonable")
else:
    print(f"⚠  w_max = {w_max_overall:.2f} m/s is outside expected range (3-8 m/s)")
    print("   Check buoyancy formulation and units")

print("="*80)
