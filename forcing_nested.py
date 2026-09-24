"""
Modified forcing functions for nested simulations.

Key difference from parent: nudge toward parent domain-mean profiles,
not toward the nested domain's own mean (which is cloud-biased).
"""

import numpy as np
import config_nested as cfg


def apply_subsidence_nested(s, parent_ref_profiles, g, dt):
    """
    Apply subsidence using parent's reference profiles.

    Same formulation as parent, but uses parent's θ₀ and qv₀ gradients.

    Parameters:
    -----------
    s : State
        Nested domain state
    parent_ref_profiles : dict
        Keys 'theta0', 'qv0' with (nz,) arrays from parent
    g : Grid
        Nested domain grid
    dt : float
        Timestep
    """
    z = g.z  # nested grid z-levels
    w_ls = -cfg.W_SUBS * np.minimum(z / cfg.Z_BL, 1.0)  # (nz,)

    # Use parent reference profiles (already at same z-levels)
    dtheta0_dz = np.gradient(parent_ref_profiles['theta0'], z)
    dqv0_dz = np.gradient(parent_ref_profiles['qv0'], z)

    # Subsidence advection: ∂φ/∂t -= w_ls · dφ₀/dz
    s.theta -= dt * w_ls[None, None, :] * dtheta0_dz[None, None, :]
    s.qv -= dt * w_ls[None, None, :] * dqv0_dz[None, None, :]


def apply_profile_nudging_nested(s, parent_mean_profiles, g, dt):
    """
    Nudge nested domain-mean toward parent domain-mean profiles.

    CRITICAL: Use parent's horizontal mean as the "environment", NOT
    the nested domain's own mean (which is cloud-biased).

    Parameters:
    -----------
    s : State
        Nested domain state
    parent_mean_profiles : dict
        Keys 'theta_mean', 'qv_mean' with (nz,) arrays
        These are the horizontal means from the full parent domain
    g : Grid
        Nested domain grid
    dt : float
        Timestep
    """
    # Nested domain-mean profiles
    nest_theta_mean = np.mean(s.theta, axis=(0, 1))  # (nz,)
    nest_qv_mean = np.mean(s.qv, axis=(0, 1))

    # Parent domain-mean profiles (the "large-scale environment")
    parent_theta_mean = parent_mean_profiles['theta_mean']
    parent_qv_mean = parent_mean_profiles['qv_mean']

    # Nudge nest mean toward parent mean
    # φ -= (dt/τ) · (φ_nest_mean - φ_parent_mean)
    s.theta -= (dt / cfg.TAU_NUDGE_T) * (
        (nest_theta_mean - parent_theta_mean)[None, None, :]
    )
    s.qv -= (dt / cfg.TAU_NUDGE_Q) * (
        (nest_qv_mean - parent_qv_mean)[None, None, :]
    )


def apply_sponge_nested(s, g, dt):
    """
    Rayleigh damping near top of nested domain.

    Same as parent, but uses nested domain's z_top.
    """
    z = g.z  # (nz,)
    z_sp = cfg.Z_SPONGE
    z_top = g.z_face[-1]

    alpha = np.where(
        z > z_sp,
        np.sin(0.5 * np.pi * (z - z_sp) / (z_top - z_sp)) ** 2 / cfg.TAU_SPONGE,
        0.0,
    )  # (nz,)

    a3 = alpha[None, None, :]  # broadcast to (1,1,nz)
    fac = np.exp(-a3 * dt)

    s.u *= fac
    s.v *= fac
    s.theta *= fac
    s.qv *= fac
    s.qc *= fac
    s.qr *= fac

    # w on faces
    alpha_face = np.zeros(g.nz + 1)
    alpha_face[1:-1] = 0.5 * (alpha[:-1] + alpha[1:])
    alpha_face[0] = alpha[0]
    alpha_face[-1] = alpha[-1]

    fac_w = np.exp(-alpha_face[None, None, :] * dt)
    s.w *= fac_w

    # Keep boundary conditions
    s.w[:, :, 0] = 0.0
    s.w[:, :, -1] = 0.0


def apply_cold_pool_nested(s, g, dt):
    """
    Cold pool parameterization (same as parent).

    Rain evaporation in sub-cloud layer - physics unchanged.
    """
    CP_DEPTH = 300.0  # m
    Lcp = cfg.LV / cfg.CP
    pi0 = s._pi0()  # (1,1,nz)

    cp_mask = g.z < CP_DEPTH  # (nz,)
    if not cp_mask.any():
        return

    qr_cp = s.qr[:, :, cp_mask]  # (nx,ny,n_cp)
    evap = np.maximum(qr_cp, 0.0)

    s.qr[:, :, cp_mask] -= evap
    s.qv[:, :, cp_mask] += evap
    s.theta[:, :, cp_mask] -= Lcp * evap / pi0[0, 0, cp_mask]

    s.qv = np.maximum(s.qv, 0.0)
    s.qr = np.maximum(s.qr, 0.0)
