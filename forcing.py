"""
Surface fluxes and sponge-layer damping.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import config as cfg
from grid import Grid
from state import State


def build_flux_map(g: Grid) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate fixed 2-D SHF and LHF maps (W/m²) with spatial correlation.
    Hot patches drive individual thermals; cool patches suppress convection.
    Called once at simulation start and reused every timestep.
    """
    rng   = np.random.default_rng(cfg.RANDOM_SEED + 1)
    sigma = cfg.SHF_CORR / g.dx     # correlation length in grid cells

    def _field():
        raw = rng.standard_normal((g.nx, g.ny))
        fld = gaussian_filter(raw, sigma=sigma)
        return fld / fld.std()

    shf_map = cfg.SHF * (1.0 + cfg.SHF_SIGMA * _field())
    lhf_map = cfg.LHF * (1.0 + cfg.SHF_SIGMA * _field())

    # Clip to physically reasonable range
    shf_map = np.maximum(shf_map, 5.0)
    lhf_map = np.maximum(lhf_map, 2.0)

    return shf_map, lhf_map


def apply_surface_fluxes(s: State, g: Grid, dt: float,
                         shf_map: np.ndarray, lhf_map: np.ndarray) -> None:
    """
    Apply spatially heterogeneous SHF and LHF into the surface cell.

    For flat domain: surface is at k=0.
    For terrain: surface follows terrain elevation (terrain-following).

    shf_map, lhf_map: (nx, ny) arrays in W/m².

    Optional cloud-radiative feedback:
    - Compute column-integrated cloud water (liquid water path, LWP)
    - Calculate optical depth: τ = extinction × LWP
    - Reduce surface fluxes by transmittance = exp(-τ)
    - Creates self-limiting clouds and oscillating thermals
    """
    # Cloud-radiative feedback (optional)
    if cfg.ENABLE_CLOUD_SHADING:
        # Liquid water path: column integral of qc [kg/m²]
        LWP = np.sum(s.qc * g.dz_face[np.newaxis, np.newaxis, :], axis=2)  # (nx, ny)

        # Cloud optical depth (dimensionless)
        # τ = extinction_coeff [m²/kg] × LWP [kg/m²]
        # Typical extinction: 100-200 m²/kg for liquid water clouds
        tau = cfg.CLOUD_EXTINCTION * LWP

        # Beer's law transmittance (0 = opaque, 1 = clear)
        transmittance = np.exp(-tau)

        # Enforce minimum transmittance (even thick clouds let some diffuse light through)
        transmittance = np.maximum(transmittance, cfg.MIN_TRANSMITTANCE)

        # Reduce fluxes under clouds
        shf_map = shf_map * transmittance
        lhf_map = lhf_map * transmittance

    if g.terrain is None:
        # ── Flat domain (original code) ──────────────────────────────────────
        rho0_bot = s.rho0[0]
        dz0      = g.dz_face[0]
        pi0_bot  = s._pi0()[0, 0, 0]

        dtheta = shf_map * dt / (rho0_bot * cfg.CP * dz0 * pi0_bot)   # (nx,ny)
        dqv    = lhf_map * dt / (rho0_bot * cfg.LV * dz0)

        s.theta[:, :, 0] += dtheta
        s.qv   [:, :, 0] += dqv

    else:
        # ── Terrain-following surface ────────────────────────────────────────
        pi0 = s._pi0()[0, 0, :]  # (nz,) - Exner function at each level

        # Apply fluxes at terrain surface level for each column
        for i in range(g.nx):
            for j in range(g.ny):
                k = g.terrain.k_sfc[i, j]  # surface level index
                dz = g.dz_face[k]
                rho = s.rho0[k]

                dtheta = shf_map[i, j] * dt / (rho * cfg.CP * dz * pi0[k])
                dqv    = lhf_map[i, j] * dt / (rho * cfg.LV * dz)

                s.theta[i, j, k] += dtheta
                s.qv[i, j, k]    += dqv


def apply_subsidence(s: State, g: Grid, dt: float) -> None:
    """
    Large-scale subsidence: linear from 0 at surface to -W_SUBS at Z_BL, constant above.
    Adds drying and adiabatic warming to θ' and qv, preventing the whole domain
    from becoming saturated over time (standard in fair-weather cumulus LES).
    """
    z    = g.z                                                     # (nz,)
    w_ls = -cfg.W_SUBS * np.minimum(z / cfg.Z_BL, 1.0)           # (nz,) negative (downward)

    dtheta0_dz = np.gradient(s.theta0, z)                         # (nz,)
    dqv0_dz    = np.gradient(s.qv0,    z)                         # (nz,)

    # subsidence advects the reference gradient: ∂φ′/∂t -= w_ls * dφ₀/dz
    s.theta -= dt * w_ls[np.newaxis, np.newaxis, :] * dtheta0_dz[np.newaxis, np.newaxis, :]
    s.qv    -= dt * w_ls[np.newaxis, np.newaxis, :] * dqv0_dz[np.newaxis, np.newaxis, :]


def apply_profile_nudging(s: State, g: Grid, dt: float) -> None:
    """
    Remove domain-mean drift of θ′ and qv toward their reference profiles.
    Represents the large-scale environment preventing unlimited thermodynamic
    drift in a periodic domain.  Only the MEAN is nudged — local variability
    driven by the heterogeneous surface is fully preserved.
    """
    # domain-mean profiles at each height
    theta_mean = np.nanmean(s.theta, axis=(0, 1))        # (nz,)
    qv_mean    = np.nanmean(s.qv,   axis=(0, 1))        # (nz,)

    # relax mean θ′ toward 0, mean qv toward qv0
    s.theta -= (dt / cfg.TAU_NUDGE_T) * theta_mean[np.newaxis, np.newaxis, :]
    s.qv    -= (dt / cfg.TAU_NUDGE_Q) * (qv_mean - s.qv0)[np.newaxis, np.newaxis, :]


def apply_cold_pool(s: State, g: Grid, dt: float) -> None:
    """
    Cold pool parameterisation.

    Rain that falls into the lowest CP_DEPTH metres is in the dry sub-cloud
    layer and evaporates rapidly.  We evaporate it instantaneously here,
    concentrating the latent cooling near the surface.  The resulting patch
    of cold, dense air spreads as a density current (handled by the existing
    pressure solver + dynamics), lifts air at its edges, and triggers new
    convection — the primary mechanism maintaining scattered cumulus.
    """
    CP_DEPTH = 300.0  # m — depth of the cold pool layer
    Lcp      = cfg.LV / cfg.CP
    pi0      = s._pi0()                        # (1,1,nz)

    cp_mask = g.z < CP_DEPTH                   # (nz,) bool — ~6 lowest levels
    if not cp_mask.any():
        return

    qr_cp = s.qr[:, :, cp_mask]               # (nx,ny,n_cp)
    evap  = np.maximum(qr_cp, 0.0)            # evaporate all rain in CP layer

    s.qr   [:, :, cp_mask] -= evap
    s.qv   [:, :, cp_mask] += evap
    s.theta[:, :, cp_mask] -= Lcp * evap / pi0[0, 0, cp_mask]

    s.qv = np.maximum(s.qv, 0.0)
    s.qr = np.maximum(s.qr, 0.0)


def apply_sponge(s: State, g: Grid, dt: float) -> None:
    """
    Rayleigh damping of all fields toward the reference state above z_sponge.
    Damping coefficient α(z) = sin²(π/2 * (z - z_sp)/(z_top - z_sp)) / τ
    """
    z     = g.z                     # (nz,)
    z_sp  = cfg.Z_SPONGE
    z_top = g.z_face[-1]

    alpha = np.where(
        z > z_sp,
        np.sin(0.5 * np.pi * (z - z_sp) / (z_top - z_sp)) ** 2 / cfg.TAU_SPONGE,
        0.0,
    )                                 # (nz,)

    a3 = alpha[np.newaxis, np.newaxis, :]   # broadcast to (1,1,nz)

    # Damp perturbations toward zero / reference
    fac = np.exp(-a3 * dt)           # smoother than 1-α*dt for large α*dt

    s.u     *= fac
    s.v     *= fac
    s.theta *= fac                   # θ′ → 0
    s.qv    *= fac                   # qv → 0 (in sponge, accuracy not critical)
    s.qc    *= fac
    s.qr    *= fac

    # w on faces: interpolate alpha to faces
    alpha_face = np.zeros(g.nz + 1)
    alpha_face[1:-1] = 0.5 * (alpha[:-1] + alpha[1:])
    alpha_face[0]    = alpha[0]
    alpha_face[-1]   = alpha[-1]

    fac_w = np.exp(-alpha_face[np.newaxis, np.newaxis, :] * dt)
    s.w   *= fac_w
    # Keep boundary w = 0
    s.w[:, :, 0]  = 0.0
    s.w[:, :, -1] = 0.0
