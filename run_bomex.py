#!/usr/bin/env python3
"""
Run BOMEX shallow cumulus intercomparison case.

This script configures the simulation for BOMEX and runs a 6-hour simulation
for validation against Siebesma et al. (2003).

Phase 2 of LES rebuild: Validation of core solver against standard case.
"""

import sys
import numpy as np
from pathlib import Path

# Modify config before importing anything else
import config as cfg

# ══════════════════════════════════════════════════════════════════════════════
# BOMEX CONFIGURATION (Override defaults)
# ══════════════════════════════════════════════════════════════════════════════

# ── Domain setup ──────────────────────────────────────────────────────────────
# BOMEX standard: 6.4 km × 6.4 km × 3 km
# Resolution: 100 m horizontal, ~40 m vertical
cfg.NX = 64           # 64 × 100m = 6.4 km
cfg.NY = 64
cfg.NZ = 75           # 3 km with exponential stretching
cfg.DX = 100.0        # m
cfg.DY = 100.0        # m
cfg.Z_TOP = 3000.0    # m (BOMEX domain)

# ── Time ──────────────────────────────────────────────────────────────────────
cfg.DT = 2.0          # s (RK3 allows larger dt)
cfg.T_END = 7200.0    # s (2 hours)
cfg.OUTPUT_EVERY = 60.0  # s (1 minute)

# ── Reference atmosphere ──────────────────────────────────────────────────────
# Will be overridden by bomex_case.initialize_bomex_state()
# But set reasonable values for grid initialization
cfg.THETA_S = 298.7   # K (BOMEX surface θ_l)
cfg.Z_BL = 520.0      # m (BOMEX well-mixed layer depth)
cfg.GAMMA_BL = 0.0    # K/km (well-mixed)
cfg.GAMMA_FT = 6.0    # K/km (inversion)
cfg.QV_SURF = 0.017   # BOMEX surface qt
cfg.QV_SCALE = 1500.0 # m (approximate scale height)

# ── Surface forcing ───────────────────────────────────────────────────────────
# BOMEX uses constant fluxes (applied in bomex_case.py)
# Disable heterogeneous forcing
cfg.SHF_SIGMA = 0.0   # No spatial variation
cfg.SHF_CORR = 1e6    # Large correlation (effectively uniform)

# ── Geostrophic wind ──────────────────────────────────────────────────────────
cfg.U_GEO = -10.0     # m/s (BOMEX westerly wind)
cfg.V_GEO = 0.0       # m/s

# ── Coriolis ──────────────────────────────────────────────────────────────────
cfg.ENABLE_CORIOLIS = True
cfg.CORIOLIS_F = 0.376e-4  # s^-1 (15°N)

# ── SGS model ─────────────────────────────────────────────────────────────────
cfg.CS = 0.18         # Smagorinsky constant (standard)
cfg.PR_T = 1.0/3.0    # Turbulent Prandtl number (standard for LES)

# ── Disable optional physics (Phase 2.2: Clean baseline) ─────────────────────
cfg.ENABLE_ENTRAINMENT = False        # LES should RESOLVE entrainment
cfg.ENABLE_DYNAMIC_ENTRAINMENT = False
cfg.ENABLE_CLOUD_SHADING = False      # No cloud-radiative feedback in standard BOMEX
cfg.ENABLE_WENO_MOMENTUM = False      # Keep momentum advection stable

# ── Disable profile nudging (BOMEX uses explicit forcing instead) ────────────
cfg.TAU_NUDGE_T = 1e10  # Effectively disable
cfg.TAU_NUDGE_Q = 1e10

# ── Subsidence disabled (will use BOMEX-specific forcing) ────────────────────
cfg.W_SUBS = 0.0  # Disable default subsidence (BOMEX has its own profile)

# ── Sponge layer ──────────────────────────────────────────────────────────────
cfg.Z_SPONGE = 2500.0   # m (start sponge above cloud layer)
cfg.TAU_SPONGE = 300.0  # s (gentler than default)

# ── Initial perturbations ─────────────────────────────────────────────────────
cfg.THETA_NOISE = 0.1   # K (small perturbations)
cfg.QV_NOISE = 1e-4     # kg/kg
cfg.NOISE_CORR = 800.0  # m

# ── No restart (fresh start for BOMEX) ───────────────────────────────────────
cfg.RESTART_FROM = None

# ══════════════════════════════════════════════════════════════════════════════
# RUN SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

print("="*70)
print("BOMEX SHALLOW CUMULUS INTERCOMPARISON")
print("="*70)
print(f"Domain: {cfg.NX}×{cfg.NY}×{cfg.NZ} ({cfg.DX}m × {cfg.DY}m × {cfg.Z_TOP}m)")
print(f"Duration: {cfg.T_END/3600:.1f} hours")
print(f"Timestep: {cfg.DT} s")
print(f"Output interval: {cfg.OUTPUT_EVERY} s")
print()
print("Phase 2: Core solver validation against Siebesma et al. (2003)")
print("="*70)
print()

# Now import simulation components (after config is modified)
from grid import Grid
from state import State
from simulation import Simulation
from bomex_case import initialize_bomex_state, apply_bomex_forcings, apply_bomex_surface_fluxes

# Create grid
print("Creating grid...")
grid = Grid()

# Create state - either from snapshot or BOMEX initialization
print("Creating state...")
if cfg.RESTART_FROM is not None:
    # Load from snapshot (restart_file parameter handles this)
    state = State(grid, restart_file=cfg.RESTART_FROM)
    print(f"Loaded state from {cfg.RESTART_FROM}")
else:
    # Fresh start with BOMEX initialization
    state = State(grid)
    print("Initializing BOMEX profiles...")
    initialize_bomex_state(state, grid)
print()

# Create simulation
print("Creating simulation...")


class BOMEXSimulation(Simulation):
    """
    BOMEX-specific simulation that uses BOMEX forcings instead of default forcings.
    """

    def __init__(self, grid: Grid, state: State):
        """Initialize BOMEX simulation with pre-configured grid and state."""
        # Don't call super().__init__() since we're providing our own grid/state
        self.grid = grid
        self.state = state

        # BOMEX uses kinematic surface fluxes (applied in apply_bomex_surface_fluxes)
        # These flux maps are not used, but rk3_step requires them as arguments
        import numpy as np
        self.shf_map = np.zeros((grid.nx, grid.ny))  # Not used for BOMEX
        self.lhf_map = np.zeros((grid.nx, grid.ny))  # Not used for BOMEX

        # Initialize time - if restarting, load from snapshot
        if cfg.RESTART_FROM is not None:
            data = np.load(cfg.RESTART_FROM)
            self.t = float(data['t'])
            # Estimate step count from time (assuming dt=2.0)
            self.step = int(self.t / cfg.DT)
        else:
            self.t = 0.0
            self.step = 0

        # Visualization and diagnostics
        from scripts.visualize.visualize import Visualizer
        import csv
        import os
        self.viz = Visualizer(self.grid)
        self.prev_centroid = None
        self.prev_t = 0.0

        # Create output directory and CSV
        os.makedirs("snapshots", exist_ok=True)
        from simulation import _DIAG_FIELDS
        # If restarting, append to existing CSV; otherwise create new one
        if cfg.RESTART_FROM is not None and os.path.exists("diag.csv"):
            self._csv = open("diag.csv", "a", newline="")
            self._writer = csv.DictWriter(self._csv, fieldnames=_DIAG_FIELDS)
            # Don't write header when appending
        else:
            self._csv = open("diag.csv", "w", newline="")
            self._writer = csv.DictWriter(self._csv, fieldnames=_DIAG_FIELDS)
            self._writer.writeheader()
        self._csv.flush()

    def run(self) -> None:
        """Override run() to properly handle restart from snapshot."""
        # Calculate next_output based on current time (for restart)
        import math
        next_output = math.ceil(self.t / cfg.OUTPUT_EVERY) * cfg.OUTPUT_EVERY
        last_dt = cfg.DT

        try:
            while self.t < cfg.T_END:
                last_dt = self._cfl_dt()
                dt = last_dt
                # Align to output times
                if self.t + dt > next_output:
                    dt = max(next_output - self.t, 1e-6)

                self._advance(dt)
                self.t += dt
                self.step += 1

                # NaN check: Fail fast on non-finite values
                if not (np.isfinite(self.state.w).all() and
                        np.isfinite(self.state.theta_l).all() and
                        np.isfinite(self.state.qt).all()):
                    raise FloatingPointError(
                        f"Non-finite state detected at t={self.t:.1f}s, step={self.step}\n"
                        f"w: min={self.state.w.min():.2e}, max={self.state.w.max():.2e}\n"
                        f"theta_l: min={self.state.theta_l.min():.2e}, max={self.state.theta_l.max():.2e}\n"
                        f"qt: min={self.state.qt.min():.2e}, max={self.state.qt.max():.2e}"
                    )

                if self.t >= next_output - 1e-6:
                    d = self._diagnostics(last_dt)
                    self._report(d)
                    self._writer.writerow({k: f"{v:.6g}" for k, v in d.items()})
                    self._csv.flush()
                    self.viz.update(self.state, self.t)
                    self._save_snapshot(d)
                    next_output += cfg.OUTPUT_EVERY
        finally:
            self._csv.close()

    def _advance(self, dt: float) -> None:
        """Override _advance() to use BOMEX-specific physics."""
        from timestepping import rk3_step
        rk3_step(self.state, self.grid, dt, self.shf_map, self.lhf_map,
                 physics_callback=self._apply_physics)

    def _apply_physics(self, s: State, g: Grid, dt: float) -> None:
        """
        Apply BOMEX-specific physics and forcings.

        Replaces the default physics from Simulation class.
        """
        # 1. BOMEX surface fluxes (constant in time and space)
        apply_bomex_surface_fluxes(s, g, dt)
        s._update_conserved_variables()  # θ_l, qt → θ, qv, qc

        # 2. BOMEX large-scale forcings (subsidence, radiative cooling, advection)
        apply_bomex_forcings(s, g, dt)
        s._update_conserved_variables()

        # 3. Microphysics (saturation adjustment only, no rain for BOMEX)
        from microphysics import saturation_adjustment
        saturation_adjustment(s, dt)
        s._update_conserved_variables()

        # 4. Sponge layer (standard, applied to all LES)
        from forcing import apply_sponge
        apply_sponge(s, g, dt)
        s._update_conserved_variables()


sim = BOMEXSimulation(grid, state)

# Run simulation
print("Starting BOMEX simulation...")
print(f"Expected completion: {cfg.T_END/3600:.1f} hours simulation time")
print()

try:
    sim.run()
    print()
    print("="*70)
    print("BOMEX SIMULATION COMPLETE")
    print("="*70)
    print(f"Output saved to: snapshots/")
    print(f"Diagnostics saved to: diag.csv")
    print()
    print("Next steps:")
    print("  1. Analyze vertical profiles: scripts/analyze_bomex.py")
    print("  2. Compare to Siebesma et al. (2003) intercomparison")
    print("  3. Generate visualizations: scripts/visualize/visualize3d.py")
    print("="*70)

except KeyboardInterrupt:
    print()
    print("="*70)
    print("SIMULATION INTERRUPTED")
    print("="*70)
    print(f"Time reached: {sim.t:.1f} s ({sim.t/3600:.2f} hours)")
    print("Partial output saved.")
    sys.exit(1)

except Exception as e:
    print()
    print("="*70)
    print("SIMULATION FAILED")
    print("="*70)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
