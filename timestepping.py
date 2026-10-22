"""
Runge-Kutta 3rd-order time integration (Wicker-Skamarock 2002).

Following LES_REBUILD_PLAN.md Phase 1.2, line 220-245.

Reference:
  Wicker, L. J., & Skamarock, W. C. (2002). Time-splitting methods for
  elastic models using forward time schemes. Monthly Weather Review, 130(8), 2088-2097.

The scheme:
  s^(1) = s^n + (dt/3) * RHS(s^n)
  s^(2) = s^n + (dt/2) * RHS(s^(1))
  s^(n+1) = s^n + dt * RHS(s^(2))

This is third-order accurate in time and more stable than forward Euler,
allowing larger CFL numbers (typically CFL ≤ 1.0 instead of ≤ 0.5).
"""

import numpy as np
import copy
from grid import Grid
from state import State
from dynamics import compute_tendencies
from pressure import project
from microphysics import apply_microphysics
from forcing import (apply_subsidence, apply_profile_nudging, apply_cold_pool, apply_sponge,
                     apply_radiative_cooling)
from surface_layer import apply_surface_layer  # Phase 1.5: M-O surface layer
from lateral_bc import apply_fringe
from open_bc import apply_relaxation as apply_open_relaxation, any_open
import config as cfg


def rk3_step(s: State, g: Grid, dt: float,
             shf_map: np.ndarray, lhf_map: np.ndarray,
             physics_callback=None, t: float = 0.0) -> None:
    """
    Advance state by one timestep using RK3 (Wicker-Skamarock 2002).

    Modifies s in-place.

    For LES with physics, we apply physics ONCE at the end, not at each substep.
    The RK3 integrates only the dynamics tendencies.

    Args:
        s: State to advance
        g: Grid
        dt: Timestep
        shf_map: Sensible heat flux map (nx, ny)
        lhf_map: Latent heat flux map (nx, ny)
        physics_callback: Optional callable (s, g, dt) -> None to replace default physics.
                         If provided, only this callback is called. If None, uses default physics.
        t: Current simulation time (for ramp-up, etc.)
    """
    # Save initial state s^n
    s0 = _copy_state(s)

    # RK3 coefficients (Wicker-Skamarock low-storage variant)
    alpha = [1.0/3.0, 1.0/2.0, 1.0]  # Coefficients for update from s0

    # ── RK3 stages (dynamics only) ────────────────────────────────────────────
    for stage in range(3):
        # Compute dynamics tendencies at current stage
        tend = compute_tendencies(s, g)

        # Update prognostic variables: s ← s0 + α[stage] * dt * RHS
        _update_state(s, s0, tend, alpha[stage] * dt)

        # Clip negative water content (WENO can produce small undershoots)
        s.qc = np.maximum(s.qc, 0.0)
        s.qr = np.maximum(s.qr, 0.0)
        s.qv = np.maximum(s.qv, 0.0)
        if 'qg' in tend:
            s.qg = np.maximum(s.qg, 0.0)

        # Pressure projection after each stage (maintains divergence-free)
        project(s.u, s.v, s.w, g, dt, rho0=s.rho0)

        # Boundary conditions after projection
        s.w[:, :, 0] = 0.0   # rigid lid
        s.w[:, :, -1] = 0.0

    # Water through open lateral boundaries: s^{n+1} = s^n + dt·RHS(stage 3)
    s.boundary_water_out += dt * s.boundary_water_rate

    # ── Apply physics ONCE at end ─────────────────────────────────────────────
    # This is more stable than applying at each substep
    #
    # Phase 1.4: Physics works on diagnostic variables (θ, qv, qc),
    # then we convert back to conserved (θ_l, qt) after each physics step.

    if physics_callback is not None:
        # Use custom physics (e.g., BOMEX-specific forcings)
        physics_callback(s, g, dt)
    else:
        # Default physics chain
        # Phase 1.5: Surface layer (M-O bulk fluxes + momentum stress)
        # Apply ramp-up factor to avoid startup shock
        ramp_factor = min(1.0, t / cfg.FLUX_RAMP_TIME) if cfg.FLUX_RAMP_TIME > 0 else 1.0
        shf_ramped = shf_map * ramp_factor
        lhf_ramped = lhf_map * ramp_factor
        apply_surface_layer(s, g, dt, shf_ramped, lhf_ramped)
        s._update_conserved_variables()  # θ, qv → θ_l, qt

        # Microphysics (saturation adjustment, warm rain, sedimentation)
        # NOTE: Microphysics modifies θ, qv, qc, qr
        apply_microphysics(s, g, dt)
        s._update_conserved_variables()  # θ, qv, qc → θ_l, qt

        # Cold pool (evaporative cooling) - controlled by ENABLE_COLD_POOL_HACK
        # Now redundant: rain sedimentation with evaporation handles this properly
        if cfg.ENABLE_COLD_POOL_HACK:
            apply_cold_pool(s, g, dt)
            s._update_conserved_variables()

        # Large-scale forcing
        apply_subsidence(s, g, dt)
        apply_profile_nudging(s, g, dt)
        apply_radiative_cooling(s, g, dt)
        s._update_conserved_variables()

        # Sponge layer (upper boundary relaxation)
        apply_sponge(s, g, dt)
        s._update_conserved_variables()

        # Inflow relaxation zone next to open lateral boundaries (no-op if periodic)
        apply_open_relaxation(s, g, dt)
        if any_open():
            s._update_conserved_variables()

    # Inflow/outflow in x: relax the fringe toward the recycled upstream strip
    if cfg.LATERAL_BC == "fringe":
        apply_fringe(s, g, dt)

    # Final boundary conditions
    _apply_boundary_conditions(s, g)

    # Final update of diagnostics
    s._update_diagnostic_variables()


def _copy_state(s: State) -> dict:
    """
    Create a lightweight copy of prognostic fields.

    Phase 1.4: Stores conserved variables θ_l, qt (not θ, qv, qc).

    Returns a dict with copies of u, v, w, theta_l, qt, qr.
    This avoids deep-copying the entire State object (which includes
    reference profiles, grid, etc.).
    """
    out = {
        'u': s.u.copy(),
        'v': s.v.copy(),
        'w': s.w.copy(),
        'theta_l': s.theta_l.copy(),
        'qt': s.qt.copy(),
        'qr': s.qr.copy(),
    }
    if getattr(cfg, 'ICE_SCHEME', 'none') == 'graupel':
        out['qg'] = s.qg.copy()
    return out


def _update_state(s: State, s0: dict, tend: dict, dt_eff: float) -> None:
    """
    Update state: s ← s0 + dt_eff * tend

    Phase 1.4: Updates conserved variables θ_l, qt.

    Args:
        s: State to update (modified in-place)
        s0: Initial state (dict from _copy_state)
        tend: Tendencies (dict from compute_tendencies)
        dt_eff: Effective timestep (α[stage] * dt)
    """
    s.u[:]       = s0['u']       + dt_eff * tend['u']
    s.v[:]       = s0['v']       + dt_eff * tend['v']
    s.w[:]       = s0['w']       + dt_eff * tend['w']
    s.theta_l[:] = s0['theta_l'] + dt_eff * tend['theta_l']
    s.qt[:]      = s0['qt']      + dt_eff * tend['qt']
    s.qr[:]      = s0['qr']      + dt_eff * tend['qr']
    if 'qg' in tend:
        s.qg[:]  = s0['qg']      + dt_eff * tend['qg']

    # Update diagnostic variables (θ, qv, qc) from conserved (θ_l, qt)
    s._update_diagnostic_variables()


def _apply_boundary_conditions(s: State, g: Grid) -> None:
    """
    Apply boundary conditions to state.

    - Bottom: Phase 1.5 - momentum handled by M-O surface layer (not hard BC)
    - Top/bottom: w=0 (rigid lid, already enforced in projection)
    - Periodic: x and y directions wrap (CRITICAL for eliminating seam artifacts!)
    - Terrain: zero out fields inside terrain mask
    - Non-negative: qv, qc, qr ≥ 0
    """
    # Phase 1.5: NO hard boundary condition at surface!
    # Surface momentum flux is applied by Monin-Obukhov surface layer
    # in apply_surface_layer() above. The artificial u=v=0 BC has been removed.
    # This allows realistic wind profiles near the surface.

    # ── Periodic boundary conditions ─────────────────────────────────────────
    # CRITICAL: Enforce after all physics/forcing steps to eliminate seam errors!
    # u is on x-faces (nx+1): u[0] and u[nx] are the SAME face
    # v is on y-faces (ny+1): v[0] and v[ny] are the SAME face
    # (only in periodic directions; open boundary faces are set by the lateral BC)
    if getattr(cfg, 'BC_X', 'periodic') != 'open':
        s.u[-1, :, :] = s.u[0, :, :]
    if getattr(cfg, 'BC_Y', 'periodic') != 'open':
        s.v[:, -1, :] = s.v[:, 0, :]

    # w boundary: rigid lid (already set in RK3 loop and projection)
    s.w[:, :, 0] = 0.0
    s.w[:, :, -1] = 0.0

    # Non-negative water (Phase 1.4: enforce on conserved variables)
    s.qt = np.maximum(s.qt, 0.0)
    s.qr = np.maximum(s.qr, 0.0)
    s.qg = np.maximum(s.qg, 0.0)

    # Terrain masking (if terrain is present)
    if g.terrain is not None:
        mask = g.terrain.mask  # (nx, ny, nz) cell-centered mask

        # Zero out scalar fields inside terrain (cell-centered)
        # Phase 1.4: Zero conserved variables
        s.theta_l[mask] = 0.0
        qt0_full = np.broadcast_to(s.qv0, (g.nx, g.ny, g.nz))  # Use qv0 as qt background
        s.qt[mask] = qt0_full[mask]
        s.qr[mask] = 0.0
        s.qg[mask] = 0.0

        # Update diagnostics after terrain masking
        s._update_diagnostic_variables()

        # u mask (x-faces, nx+1): mask if either adjacent cell is terrain
        mask_u = np.zeros((g.nx+1, g.ny, g.nz), dtype=bool)
        mask_u[1:-1, :, :] = mask[:-1, :, :] | mask[1:, :, :]  # interior faces
        mask_u[0, :, :] = mask[0, :, :]    # left boundary
        mask_u[-1, :, :] = mask[-1, :, :]  # right boundary (periodic)
        s.u[mask_u] = 0.0

        # v mask (y-faces, ny+1): mask if either adjacent cell is terrain
        mask_v = np.zeros((g.nx, g.ny+1, g.nz), dtype=bool)
        mask_v[:, 1:-1, :] = mask[:, :-1, :] | mask[:, 1:, :]  # interior faces
        mask_v[:, 0, :] = mask[:, 0, :]    # bottom boundary
        mask_v[:, -1, :] = mask[:, -1, :]  # top boundary (periodic)
        s.v[mask_v] = 0.0

        # w mask (z-faces, nz+1): mask if cell below or above is terrain
        mask_w_lower = np.concatenate([mask[:, :, :1], mask], axis=2)  # (nx,ny,nz+1)
        mask_w_upper = np.concatenate([mask, mask[:, :, -1:]], axis=2)  # (nx,ny,nz+1)
        mask_w = mask_w_lower | mask_w_upper
        s.w[mask_w] = 0.0
