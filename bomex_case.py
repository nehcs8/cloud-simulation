"""
BOMEX shallow cumulus case setup.

Reference: Siebesma et al. (2003) - A large eddy simulation intercomparison
study of shallow cumulus convection. J. Atmos. Sci.

Standard case for validating shallow cumulus LES models.
"""

import numpy as np
import config as cfg
from grid import Grid
from state import State


# ══════════════════════════════════════════════════════════════════════════════
# BOMEX REFERENCE PROFILES
# ══════════════════════════════════════════════════════════════════════════════

def bomex_theta_l_profile(z: np.ndarray) -> np.ndarray:
    """
    BOMEX liquid water potential temperature profile θ_l(z) [K].

    Reference sounding from Siebesma et al. (2003):
    z [m]     θ_l [K]
    0         298.7
    520       298.7     (well-mixed BL)
    1480      302.4     (cloud layer top)
    2000      308.2     (inversion!)
    3000      311.85    (free troposphere)
    """
    theta_l = np.zeros_like(z)

    # Piecewise linear interpolation
    # Below 520m: constant (well-mixed)
    mask1 = z <= 520.0
    theta_l[mask1] = 298.7

    # 520m to 1480m: gradual increase
    mask2 = (z > 520.0) & (z <= 1480.0)
    theta_l[mask2] = 298.7 + (302.4 - 298.7) * (z[mask2] - 520.0) / (1480.0 - 520.0)

    # 1480m to 2000m: sharp inversion
    mask3 = (z > 1480.0) & (z <= 2000.0)
    theta_l[mask3] = 302.4 + (308.2 - 302.4) * (z[mask3] - 1480.0) / (2000.0 - 1480.0)

    # Above 2000m: free troposphere
    mask4 = z > 2000.0
    theta_l[mask4] = 308.2 + (311.85 - 308.2) * (z[mask4] - 2000.0) / (3000.0 - 2000.0)

    return theta_l


def bomex_qt_profile(z: np.ndarray) -> np.ndarray:
    """
    BOMEX total water mixing ratio profile qt(z) [kg/kg].

    Reference sounding from Siebesma et al. (2003):
    z [m]     qt [g/kg]
    0         17.0
    520       16.3      (well-mixed BL)
    1480      10.7      (cloud layer top)
    2000      4.2       (dry free troposphere!)
    3000      3.0
    """
    qt = np.zeros_like(z)

    # Convert g/kg to kg/kg
    # Below 520m: nearly constant (well-mixed)
    mask1 = z <= 520.0
    qt[mask1] = 0.017 - (0.017 - 0.0163) * z[mask1] / 520.0

    # 520m to 1480m: gradual decrease
    mask2 = (z > 520.0) & (z <= 1480.0)
    qt[mask2] = 0.0163 + (0.0107 - 0.0163) * (z[mask2] - 520.0) / (1480.0 - 520.0)

    # 1480m to 2000m: sharp drop (dry air intrusion)
    mask3 = (z > 1480.0) & (z <= 2000.0)
    qt[mask3] = 0.0107 + (0.0042 - 0.0107) * (z[mask3] - 1480.0) / (2000.0 - 1480.0)

    # Above 2000m: dry free troposphere
    mask4 = z > 2000.0
    qt[mask4] = 0.0042 + (0.0030 - 0.0042) * (z[mask4] - 2000.0) / (3000.0 - 2000.0)

    return qt


# ══════════════════════════════════════════════════════════════════════════════
# LARGE-SCALE FORCINGS
# ══════════════════════════════════════════════════════════════════════════════

def bomex_subsidence(z: np.ndarray) -> np.ndarray:
    """
    BOMEX large-scale subsidence w_ls(z) [m/s].

    Subsidence profile:
    - z < 1500m: w_ls = -0.0065 * (z / 1500)  m/s  (linear)
    - z >= 1500m: w_ls = -0.0065 m/s  (constant)

    Negative indicates downward motion (compressing the boundary layer).
    """
    w_ls = np.zeros_like(z)

    mask_below = z < 1500.0
    w_ls[mask_below] = -0.0065 * z[mask_below] / 1500.0

    mask_above = z >= 1500.0
    w_ls[mask_above] = -0.0065

    return w_ls


def bomex_advective_tendency_qt(z: np.ndarray) -> np.ndarray:
    """
    BOMEX large-scale advective drying tendency [s^-1].

    From Siebesma et al. (2003) Table B1:
    ∂qt/∂t|_adv = -1.2e-8 s^-1  for z < 300m
    ∂qt/∂t|_adv = 0             for z >= 300m

    CRITICAL FIX: Changed from 1500m to 300m to match published specification.
    This represents horizontal advection bringing in drier air near the surface.
    """
    dqt_dt_adv = np.zeros_like(z)

    mask = z < 300.0  # CRITICAL: 300m, not 1500m!
    dqt_dt_adv[mask] = -1.2e-8  # s^-1

    return dqt_dt_adv


def bomex_radiative_cooling() -> float:
    """
    BOMEX radiative cooling rate [K/s].

    Constant cooling of -2 K/day throughout the troposphere.
    """
    return -2.0 / 86400.0  # K/day → K/s


# ══════════════════════════════════════════════════════════════════════════════
# SURFACE FLUXES
# ══════════════════════════════════════════════════════════════════════════════

def bomex_surface_fluxes():
    """
    BOMEX surface fluxes (constant in time and space).

    From Siebesma et al. (2003):
      Sensible heat flux: <w'θ_l'> = 8×10^-3 K·m/s (kinematic)
      Latent heat flux: <w'q_t'> = 5.2×10^-5 kg/kg·m/s (kinematic)

    These are KINEMATIC fluxes, not W/m².
    To convert to W/m² for reference:
      SHF = ρ * c_p * w'θ' ≈ 1.2 kg/m³ * 1004 J/kg/K * 0.008 K·m/s ≈ 9.6 W/m²
      LHF = ρ * L_v * w'q' ≈ 1.2 kg/m³ * 2.5e6 J/kg * 5.2e-5 ≈ 156 W/m²

    Returns:
        w'θ_l'_surface: 8e-3 K·m/s  (kinematic heat flux)
        w'qt'_surface: 5.2e-5 kg/kg·m/s  (kinematic moisture flux)
    """
    wthl_surface = 8e-3       # K·m/s (kinematic)
    wqt_surface = 5.2e-5      # kg/kg·m/s (kinematic)
    return wthl_surface, wqt_surface


# ══════════════════════════════════════════════════════════════════════════════
# GEOSTROPHIC WIND AND CORIOLIS
# ══════════════════════════════════════════════════════════════════════════════

def bomex_geostrophic_wind():
    """
    BOMEX geostrophic wind (constant with height).

    Returns:
        u_g: -10 m/s  (westerly wind)
        v_g: 0 m/s
        f: 0.376e-4 s^-1  (Coriolis parameter at 15°N)
    """
    u_g = -10.0           # m/s
    v_g = 0.0             # m/s
    f = 0.376e-4          # s^-1 (15°N)
    return u_g, v_g, f


# ══════════════════════════════════════════════════════════════════════════════
# APPLY BOMEX FORCINGS
# ══════════════════════════════════════════════════════════════════════════════

def apply_bomex_forcings(s: State, g: Grid, dt: float) -> None:
    """
    Apply all BOMEX large-scale forcings to the state.

    Forcings include:
    1. Subsidence: advects mean state vertically
    2. Radiative cooling: -2 K/day
    3. Large-scale advection: drying below 1500m
    4. Coriolis force: drives flow toward geostrophic balance (applied in dynamics)

    Note: Surface fluxes are handled by surface_layer.py
    Note: Coriolis force is handled in dynamics.py
    """
    z = g.z

    # ── 1. Subsidence advection ───────────────────────────────────────────────
    # CRITICAL FIX: Subsidence advects TOTAL profile (reference + perturbation)
    # ∂φ'/∂t = -w_ls * ∂(φ₀ + <φ'>)/∂z
    #
    # For BOMEX, both θ_l and q_t have strong gradients in the reference profile
    # (especially across the inversion at 1480-2000m). Neglecting dθ₀/dz would
    # cause spurious warming in the free troposphere.

    w_ls = bomex_subsidence(z)  # (nz,)

    # Compute domain-mean profiles
    theta_l_mean = np.mean(s.theta_l, axis=(0, 1))  # (nz,) - perturbation mean
    qt_mean = np.mean(s.qt, axis=(0, 1))            # (nz,) - total mean

    # Gradients of reference + mean perturbation
    # theta_l is stored as PERTURBATION, so add reference profile first
    theta_l_total_mean = s.theta0 + theta_l_mean  # (nz,) - total θ_l profile
    dtheta_l_total_dz = np.gradient(theta_l_total_mean, z)  # (nz,)

    # qt is stored as TOTAL, so gradient directly
    dqt_mean_dz = np.gradient(qt_mean, z)  # (nz,)

    # Apply subsidence tendency: ∂φ'/∂t = -w_ls * ∂(φ₀ + <φ'>)/∂z
    # This affects the PERTURBATION field (s.theta_l)
    subsidence_theta_l = -w_ls * dtheta_l_total_dz
    subsidence_qt = -w_ls * dqt_mean_dz

    s.theta_l += dt * subsidence_theta_l[np.newaxis, np.newaxis, :]
    s.qt += dt * subsidence_qt[np.newaxis, np.newaxis, :]

    # ── 2. Radiative cooling ──────────────────────────────────────────────────
    # Constant -2 K/day throughout the domain
    dtheta_dt_rad = bomex_radiative_cooling()  # K/s

    # Apply to theta_l (affects temperature directly)
    s.theta_l += dt * dtheta_dt_rad

    # ── 3. Large-scale advective drying ───────────────────────────────────────
    # Horizontal advection brings in drier air below 1500m
    dqt_dt_adv = bomex_advective_tendency_qt(z)  # (nz,)

    s.qt += dt * dqt_dt_adv[np.newaxis, np.newaxis, :]


def apply_bomex_surface_fluxes(s: State, g: Grid, dt: float) -> None:
    """
    Apply BOMEX surface fluxes (constant in time and space).

    Uses kinematic fluxes (simpler than M-O theory for this idealized case).
    """
    wthl_surface, wqt_surface = bomex_surface_fluxes()

    # Surface fluxes are kinematic: w'φ' [φ·m/s]
    # Flux into surface cell: ∂φ/∂t = w'φ'_surface / dz

    dz0 = g.dz_face[0]  # Height of surface cell

    # Apply to surface cell (k=0)
    s.theta_l[:, :, 0] += dt * wthl_surface / dz0
    s.qt[:, :, 0] += dt * wqt_surface / dz0


# ══════════════════════════════════════════════════════════════════════════════
# INITIALIZE BOMEX STATE
# ══════════════════════════════════════════════════════════════════════════════

def initialize_bomex_state(s: State, g: Grid) -> None:
    """
    Initialize state with BOMEX reference profiles.

    Overwrites the default initialization to use BOMEX sounding.
    Call this AFTER State.__init__() to replace default profiles.
    """
    z = g.z  # Cell center heights

    # ── Reference profiles ────────────────────────────────────────────────────
    theta_l_ref = bomex_theta_l_profile(z)  # (nz,)
    qt_ref = bomex_qt_profile(z)            # (nz,)

    # BOMEX uses theta_l as the base state (not standard theta)
    # Our code expects theta0 as base state, so we need to be careful
    # For now, set theta0 = theta_l_ref (no clouds initially, so theta = theta_l)
    s.theta0[:] = theta_l_ref
    s.qv0[:] = qt_ref  # Initially all vapor, no cloud

    # ── Initialize prognostic fields ──────────────────────────────────────────
    # Start with reference profiles + small perturbations
    s.theta_l[:] = 0.0  # Perturbation (theta_l = theta_l_ref + theta_l')
    s.qt[:] = qt_ref[np.newaxis, np.newaxis, :]
    s.qr[:] = 0.0  # No rain initially

    # Add random perturbations in boundary layer (same as before)
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    sigma = cfg.NOISE_CORR / g.dx

    def _corr_noise():
        raw = rng.standard_normal((g.nx, g.ny))
        fld = gaussian_filter(raw, sigma=sigma)
        return fld / fld.std()

    theta_l_noise_2d = _corr_noise()
    qt_noise_2d = _corr_noise()

    # Vertical taper (full amplitude at surface, zero at BL top)
    z_bl_bomex = 520.0  # BOMEX well-mixed layer depth
    taper = np.maximum(0.0, 1.0 - z / z_bl_bomex) ** 2

    # Apply perturbations
    s.theta_l += cfg.THETA_NOISE * theta_l_noise_2d[:, :, np.newaxis] * taper
    s.qt += cfg.QV_NOISE * qt_noise_2d[:, :, np.newaxis] * taper

    # ── Initialize winds ──────────────────────────────────────────────────────
    u_g, v_g, f = bomex_geostrophic_wind()

    # Initialize u, v to geostrophic wind
    # u, v are face-staggered in C-grid
    s.u[:] = u_g
    s.v[:] = v_g
    s.w[:] = 0.0  # No initial vertical motion

    # ── Update diagnostic variables ───────────────────────────────────────────
    # Convert theta_l, qt → theta, qv, qc
    s._update_diagnostic_variables()

    print("="*70)
    print("BOMEX INITIALIZATION")
    print("="*70)
    print(f"Domain: {g.nx}×{g.ny}×{g.nz} = {g.dx}×{g.dy}m × {g.z[-1]:.0f}m height")
    print(f"\nReference profiles:")
    print(f"  Surface θ_l: {theta_l_ref[0]:.1f} K")
    print(f"  Cloud base θ_l: {theta_l_ref[np.argmin(np.abs(z - 520))]:.1f} K")
    print(f"  Cloud top θ_l: {theta_l_ref[np.argmin(np.abs(z - 1480))]:.1f} K")
    print(f"  Inversion θ_l: {theta_l_ref[np.argmin(np.abs(z - 2000))]:.1f} K")
    print(f"\n  Surface qt: {qt_ref[0]*1e3:.1f} g/kg")
    print(f"  Cloud top qt: {qt_ref[np.argmin(np.abs(z - 1480))]*1e3:.1f} g/kg")
    print(f"  Above inversion qt: {qt_ref[np.argmin(np.abs(z - 2000))]*1e3:.1f} g/kg")
    print(f"\nGeostrophic wind: u_g={u_g:.1f} m/s, v_g={v_g:.1f} m/s")
    print(f"Coriolis parameter: f={f:.2e} s⁻¹ (15°N)")
    print(f"\nSurface fluxes:")
    wthl, wqt = bomex_surface_fluxes()
    print(f"  w'θ_l': {wthl*1e3:.1f} mK·m/s")
    print(f"  w'qt': {wqt*1e6:.1f} mg/kg·m/s")
    print("="*70)
