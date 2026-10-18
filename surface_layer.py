"""
Monin-Obukhov surface layer physics (Phase 1.5).

Following LES_REBUILD_PLAN.md line 303-334:
  "Implement Monin-Obukhov surface layer"
  "Apply surface stress u*² as momentum flux (not velocity BC)"

Replaces the artificial u=v=0 wall boundary condition with proper
surface-layer physics based on similarity theory.

Physical model:
  - Surface friction velocity u*
  - Monin-Obukhov length L (stability parameter)
  - Stability functions ψ_m, ψ_h (Businger-Dyer)
  - Bulk flux formulation for momentum, heat, moisture

References:
  - Businger et al. (1971) - Flux-profile relationships in the atmospheric surface layer
  - Paulson (1970) - Mathematical representation of wind and temperature profiles
  - Sullivan et al. (1994) - Large-eddy simulation of the convective boundary layer
  - Moeng (1984) - A large-eddy-simulation model for the study of atmospheric turbulence
"""

import numpy as np
import config as cfg


# Von Karman constant
KAPPA = 0.4

# Roughness lengths (typical values for flat terrain)
Z0_M = 0.01   # m, momentum roughness length (grass/crops)
Z0_H = 0.001  # m, heat/moisture roughness length (typically Z0_H ≈ 0.1 * Z0_M)


def stability_functions(zeta: np.ndarray) -> tuple:
    """
    Businger-Dyer stability functions for momentum and heat.

    Uses the Businger et al. (1971) and Paulson (1970) formulations,
    which are standard in atmospheric LES.

    Args:
        zeta: z/L, stability parameter (dimensionless)
              zeta > 0: stable (suppresses turbulence)
              zeta < 0: unstable (enhances turbulence)
              zeta = 0: neutral

    Returns:
        psi_m: Integrated similarity function for momentum
        psi_h: Integrated similarity function for heat
    """
    # Clip extreme stability to avoid numerical issues
    zeta = np.clip(zeta, -10.0, 10.0)

    # Initialize arrays
    psi_m = np.zeros_like(zeta)
    psi_h = np.zeros_like(zeta)

    # Unstable conditions (zeta < 0)
    unstable = zeta < 0
    if np.any(unstable):
        x = (1.0 - 16.0 * zeta[unstable]) ** 0.25  # Businger et al. (1971)
        # Paulson (1970) integrated forms
        psi_m[unstable] = (2.0 * np.log((1.0 + x) / 2.0) +
                           np.log((1.0 + x**2) / 2.0) -
                           2.0 * np.arctan(x) + np.pi / 2.0)
        psi_h[unstable] = 2.0 * np.log((1.0 + x**2) / 2.0)

    # Stable conditions (zeta > 0)
    stable = zeta > 0
    if np.any(stable):
        # Businger et al. (1971) linear form for stable
        psi_m[stable] = -5.0 * zeta[stable]
        psi_h[stable] = -5.0 * zeta[stable]

    # Neutral (zeta = 0): psi = 0 (already initialized)

    return psi_m, psi_h


def compute_friction_velocity(u_lowest: np.ndarray, v_lowest: np.ndarray,
                               z_lowest: float, theta_lowest: np.ndarray,
                               theta_surface: np.ndarray,
                               max_iterations: int = 5) -> tuple:
    """
    Compute surface friction velocity u* using iterative bulk flux method.

    Given wind speed at lowest model level, iteratively solve for u* and L.

    Args:
        u_lowest: Horizontal wind u at lowest level [m/s] (nx, ny)
        v_lowest: Horizontal wind v at lowest level [m/s] (nx, ny)
        z_lowest: Height of lowest level above surface [m] (scalar)
        theta_lowest: Potential temperature at lowest level [K] (nx, ny)
        theta_surface: Surface potential temperature [K] (nx, ny)
        max_iterations: Maximum iterations for u* calculation

    Returns:
        u_star: Friction velocity [m/s] (nx, ny)
        L: Monin-Obukhov length [m] (nx, ny)
        theta_star: Temperature scale [K] (nx, ny)
    """
    # Wind speed magnitude at lowest level
    u_mag = np.sqrt(u_lowest**2 + v_lowest**2)
    u_mag = np.maximum(u_mag, 0.1)  # Minimum wind speed to avoid division by zero

    # Potential temperature difference (positive = surface warmer than air)
    dtheta = theta_surface - theta_lowest

    # Initial guess: neutral conditions (L → ∞, ψ = 0)
    # Neutral drag law: u* = κ * u / ln(z/z0)
    u_star = KAPPA * u_mag / np.log(z_lowest / Z0_M)
    theta_star = KAPPA * dtheta / np.log(z_lowest / Z0_H)

    # Iterative refinement accounting for stability
    for iteration in range(max_iterations):
        # Monin-Obukhov length
        # L = -u*³ * θ / (κ * g * θ*)
        # Negative for unstable (surface heating), positive for stable
        with np.errstate(divide='ignore', invalid='ignore'):
            L = -u_star**3 * theta_lowest / (KAPPA * cfg.G * theta_star)
            L = np.where(np.abs(theta_star) < 1e-6, 1e6, L)  # Neutral limit

        # Stability parameter
        zeta_m = z_lowest / L
        zeta_h = z_lowest / L

        # Stability functions
        psi_m, psi_h = stability_functions(zeta_m)

        # Updated u* and θ* with stability correction
        u_star_new = KAPPA * u_mag / (np.log(z_lowest / Z0_M) - psi_m)
        theta_star_new = KAPPA * dtheta / (np.log(z_lowest / Z0_H) - psi_h)

        # Check convergence
        du_star = np.max(np.abs(u_star_new - u_star))
        if du_star < 1e-4:
            break

        u_star = u_star_new
        theta_star = theta_star_new

    # Final L with converged u*, θ*
    with np.errstate(divide='ignore', invalid='ignore'):
        L = -u_star**3 * theta_lowest / (KAPPA * cfg.G * theta_star)
        L = np.where(np.abs(theta_star) < 1e-6, 1e6, L)

    return u_star, L, theta_star


def apply_surface_layer(s, g, dt: float, shf_map: np.ndarray, lhf_map: np.ndarray) -> None:
    """
    Apply Monin-Obukhov surface layer physics.

    Replaces the artificial u=v=0 boundary condition with proper bulk flux
    formulation. Computes surface stress, heat flux, and moisture flux based
    on similarity theory.

    Args:
        s: State object
        g: Grid object
        dt: Timestep [s]
        shf_map: Sensible heat flux map [W/m²] (nx, ny)
        lhf_map: Latent heat flux map [W/m²] (nx, ny)

    Physics applied:
        - Momentum flux: τ = ρ * u*²
        - Heat flux: w'θ' = u* * θ*
        - Moisture flux: w'q' = u* * q*
    """
    # Optional cloud shading: reduce fluxes under clouds (Beer's law on LWP)
    if cfg.ENABLE_CLOUD_SHADING:
        from forcing import cloud_transmittance
        transmittance = cloud_transmittance(s, g)
        shf_map = shf_map * transmittance
        lhf_map = lhf_map * transmittance

    if g.terrain is not None:
        _apply_surface_layer_terrain(s, g, dt, shf_map, lhf_map)
        return

    # Lowest model level (k=0 for flat terrain)
    k = 0
    z_lowest = g.z[k]  # Height above surface
    dz = g.dz_face[k]
    rho_sfc = s.rho0[k]

    # Interpolate face velocities to cell centers at lowest level
    u_lowest = 0.5 * (s.u[:-1, :, k] + s.u[1:, :, k])  # (nx, ny)
    v_lowest = 0.5 * (s.v[:, :-1, k] + s.v[:, 1:, k])  # (nx, ny)

    # Potential temperature at lowest level and surface
    # theta_lowest is already a perturbation, add background
    theta_abs_lowest = s.theta[:, :, k] + s.theta0[k]  # (nx, ny)

    # Surface temperature from sensible heat flux
    # SHF = ρ * cp * u* * θ*  →  θ* = SHF / (ρ * cp * u*)
    # For now, assume surface is slightly warmer than lowest level
    # Better: derive from specified SHF (done below)

    # Compute friction velocity iteratively
    # For heat flux: θ* = -w'θ' / u* = -SHF / (ρ * cp * u*)
    # We'll use SHF to compute θ* after getting u*

    # Initial guess for θ_surface based on SHF
    # Assume moderate u* ~ 0.3 m/s for initial guess
    u_star_guess = 0.3
    theta_star_guess = shf_map / (rho_sfc * cfg.CP * u_star_guess)
    theta_surface = theta_abs_lowest + theta_star_guess * np.log(z_lowest / Z0_H) / KAPPA

    # Iteratively compute u*, L, θ*
    u_star, L, theta_star = compute_friction_velocity(
        u_lowest, v_lowest, z_lowest, theta_abs_lowest, theta_surface
    )

    # Refine θ_surface using computed u*
    theta_star = shf_map / (rho_sfc * cfg.CP * u_star + 1e-6)
    theta_surface = theta_abs_lowest + theta_star * np.log(z_lowest / Z0_H) / KAPPA

    # Similarly for moisture
    q_star = lhf_map / (rho_sfc * cfg.LV * u_star + 1e-6)

    # Apply momentum flux (surface stress)
    # τ = ρ * u*² in direction of wind
    # du/dt = -τ_x / (ρ * Δz),  dv/dt = -τ_y / (ρ * Δz)
    u_mag = np.sqrt(u_lowest**2 + v_lowest**2) + 1e-6
    tau_x = rho_sfc * u_star**2 * (u_lowest / u_mag)  # (nx, ny)
    tau_y = rho_sfc * u_star**2 * (v_lowest / u_mag)  # (nx, ny)

    # Apply stress as tendency (not BC!)
    # On C-grid: u is on x-faces, v is on y-faces
    # We need to map cell-center stress to face locations

    # For u-faces: average tau_x from neighboring cells
    tau_x_faces = np.zeros((g.nx + 1, g.ny))
    tau_x_faces[1:-1, :] = 0.5 * (tau_x[:-1, :] + tau_x[1:, :])
    tau_x_faces[0, :] = tau_x[0, :]    # periodic
    tau_x_faces[-1, :] = tau_x[-1, :]

    # For v-faces: average tau_y from neighboring cells
    tau_y_faces = np.zeros((g.nx, g.ny + 1))
    tau_y_faces[:, 1:-1] = 0.5 * (tau_y[:, :-1] + tau_y[:, 1:])
    tau_y_faces[:, 0] = tau_y[:, 0]    # periodic
    tau_y_faces[:, -1] = tau_y[:, -1]

    # Apply stress as flux divergence
    # du/dt = -dτ/dz ≈ -τ/Δz (at surface)
    s.u[:, :, k] -= dt * tau_x_faces / (rho_sfc * dz)
    s.v[:, :, k] -= dt * tau_y_faces / (rho_sfc * dz)

    # Apply heat flux (already done via SHF in forcing.py, but keep consistent)
    # w'θ' = -u* * θ*  (sign convention: positive upward flux)
    # dθ/dt = -d(w'θ')/dz ≈ w'θ' / Δz at surface
    pi0_bot = s._pi0()[0, 0, k]
    dtheta = u_star * theta_star * dt / (dz * pi0_bot)
    s.theta[:, :, k] += dtheta

    # Apply moisture flux
    # w'q' = -u* * q*
    dqv = u_star * q_star * dt / dz
    s.qv[:, :, k] += dqv


def _apply_surface_layer_terrain(s, g, dt: float, shf_map: np.ndarray, lhf_map: np.ndarray) -> None:
    """
    apply_surface_layer on the terrain surface: each column uses its first fluid level
    k_sfc (height above ground Δz/2 of that level); the stress on a u- (v-) face acts at the
    face's lowest fluid level, the higher k_sfc of its two neighbour cells.
    """
    ter = g.terrain
    kk = ter.k_sfc                                         # (nx, ny)
    nx, ny = g.nx, g.ny
    I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    hcg = ter.hc[I, J, kk]                                 # fluid fraction of the ground cell
    # height of the fluid centre above ground; thickness of the fluid layer (partial cells: hc·Δz)
    z_lowest = np.where(hcg < 1.0, 0.5 * hcg * g.dz_face[kk], g.z[kk] - g.z_face[kk])
    dz = hcg * g.dz_face[kk]
    rho_sfc = s.rho0[kk]

    u_lowest = 0.5 * (s.u[:-1][I, J, kk] + s.u[1:][I, J, kk])
    v_lowest = 0.5 * (s.v[:, :-1][I, J, kk] + s.v[:, 1:][I, J, kk])
    theta_abs_lowest = s.theta[I, J, kk] + s.theta0[kk]

    u_star_guess = 0.3
    theta_star_guess = shf_map / (rho_sfc * cfg.CP * u_star_guess)
    theta_surface = theta_abs_lowest + theta_star_guess * np.log(z_lowest / Z0_H) / KAPPA
    u_star, L, theta_star = compute_friction_velocity(
        u_lowest, v_lowest, z_lowest, theta_abs_lowest, theta_surface)
    theta_star = shf_map / (rho_sfc * cfg.CP * u_star + 1e-6)
    q_star = lhf_map / (rho_sfc * cfg.LV * u_star + 1e-6)

    u_mag = np.sqrt(u_lowest**2 + v_lowest**2) + 1e-6
    tau_x = rho_sfc * u_star**2 * (u_lowest / u_mag)
    tau_y = rho_sfc * u_star**2 * (v_lowest / u_mag)

    open_x = getattr(cfg, 'BC_X', 'periodic') == 'open'
    open_y = getattr(cfg, 'BC_Y', 'periodic') == 'open'
    # u-faces: stress averaged from the two neighbour cells, applied at the face's ground level
    tx = np.zeros((nx + 1, ny)); kx = np.zeros((nx + 1, ny), dtype=int)
    tx[1:-1] = 0.5 * (tau_x[:-1] + tau_x[1:]); kx[1:-1] = np.maximum(kk[:-1], kk[1:])
    if open_x:
        tx[0], tx[-1], kx[0], kx[-1] = tau_x[0], tau_x[-1], kk[0], kk[-1]
    else:
        tx[0] = tx[-1] = 0.5 * (tau_x[-1] + tau_x[0]); kx[0] = kx[-1] = np.maximum(kk[-1], kk[0])
    ty = np.zeros((nx, ny + 1)); ky = np.zeros((nx, ny + 1), dtype=int)
    ty[:, 1:-1] = 0.5 * (tau_y[:, :-1] + tau_y[:, 1:]); ky[:, 1:-1] = np.maximum(kk[:, :-1], kk[:, 1:])
    if open_y:
        ty[:, 0], ty[:, -1], ky[:, 0], ky[:, -1] = tau_y[:, 0], tau_y[:, -1], kk[:, 0], kk[:, -1]
    else:
        ty[:, 0] = ty[:, -1] = 0.5 * (tau_y[:, -1] + tau_y[:, 0]); ky[:, 0] = ky[:, -1] = np.maximum(kk[:, -1], kk[:, 0])
    Iu, Ju = np.meshgrid(np.arange(nx + 1), np.arange(ny), indexing='ij')
    Iv, Jv = np.meshgrid(np.arange(nx), np.arange(ny + 1), indexing='ij')
    # stress over the open part of the face (aperture × Δz; 1 for the staircase)
    import pressure
    rho0_w = np.empty(g.nz + 1); rho0_w[0], rho0_w[-1] = s.rho0[0], s.rho0[-1]
    rho0_w[1:-1] = 0.5 * (s.rho0[:-1] + s.rho0[1:])
    t = pressure._terrain_setup(g, s.rho0, rho0_w, open_x, open_y)
    ax_f, ay_f = t['ax'][Iu, Ju, kx], t['ay'][Iv, Jv, ky]
    s.u[Iu, Ju, kx] -= dt * tx / (s.rho0[kx] * np.where(ax_f > 0, ax_f, 1.0) * g.dz_face[kx])
    s.v[Iv, Jv, ky] -= dt * ty / (s.rho0[ky] * np.where(ay_f > 0, ay_f, 1.0) * g.dz_face[ky])

    pi0 = s._pi0()[0, 0]
    s.theta[I, J, kk] += u_star * theta_star * dt / (dz * pi0[kk])
    s.qv[I, J, kk] += u_star * q_star * dt / dz
