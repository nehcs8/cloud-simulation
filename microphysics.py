"""
Kessler warm-rain bulk microphysics.

Processes (applied in order each timestep):
  1. Saturation adjustment  – condensation / evaporation of cloud water
  2. Autoconversion         – cloud water → rain  (Kessler 1969)
  3. Accretion              – rain collecting cloud droplets
  4. Rain evaporation       – sub-saturated air evaporates rain
  5. Sedimentation          – rain falls at terminal velocity
"""

import numpy as np
import config as cfg
import thermodynamics
from grid import Grid
from state import State

try:
    import kernels
except ImportError:          # numba not installed → numpy path only
    kernels = None


def _numba() -> bool:
    return kernels is not None and getattr(cfg, 'USE_NUMBA', True)


def _profiles(s: State):
    """Contiguous float64 copies of θ, qv, qc, qr and (θ₀, π₀, p) profiles for the kernels."""
    pi0 = np.ascontiguousarray(s._pi0().reshape(-1), dtype=np.float64)
    p = cfg.P00 * pi0 ** (cfg.CP / cfg.RD)
    return np.asarray(s.theta0, dtype=np.float64), pi0, p


# ── Saturation mixing ratio ───────────────────────────────────────────────────

def _q_sat(T: np.ndarray, pi0: np.ndarray) -> np.ndarray:
    """
    Saturation water-vapour mixing ratio [kg/kg].
    Uses Buck (1981) approximation for e_s(T) [Pa].
    T in K, pi0 dimensionless Exner function → p = p00 * pi0^(cp/Rd).
    """
    T_C = T - 273.15
    e_s = 611.2 * np.exp(17.67 * T_C / (T_C + 243.5))   # Pa
    p   = cfg.P00 * pi0 ** (cfg.CP / cfg.RD)
    return cfg.EPS * e_s / np.maximum(p - e_s, 1.0)       # kg/kg


# ── Saturation adjustment ─────────────────────────────────────────────────────

def saturation_adjustment(s: State, dt: float) -> None:
    """
    Adjust θ′, qv, qc so the parcel is exactly at saturation when
    qv > q_sat.  Uses an iterative Newton step (2 iterations sufficient).
    Modifies s.theta, s.qv, s.qc in-place.
    ICE_SCHEME "mixed": exact mixed-phase adjustment via θ_l (thermodynamics.diagnose_cloud_water_mixed).
    """
    if thermodynamics._mixed():
        pi0 = s._pi0()
        theta_l = thermodynamics.theta_l_from_state(s.theta, s.qc, pi0, s.theta0)
        s.qv, s.qc, s.theta = thermodynamics.diagnose_cloud_water_mixed(theta_l, s.qv + s.qc, s.theta0, pi0)
        return
    if _numba():
        theta0, pi0_1d, p = _profiles(s)
        s.theta, s.qv, s.qc = (np.array(a, dtype=np.float64) for a in (s.theta, s.qv, s.qc))
        kernels.saturation_adjustment(s.theta, s.qv, s.qc, theta0, pi0_1d, p,
                                      cfg.LV / cfg.CP, cfg.LV, cfg.RV, cfg.EPS)
        return
    pi0 = s._pi0()                                         # (1,1,nz)
    theta_abs = s.theta + s.theta0[np.newaxis, np.newaxis, :]
    T   = theta_abs * pi0                                  # (nx,ny,nz)
    q_s = _q_sat(T, pi0)

    excess = s.qv - q_s                                    # positive → condense

    # --- condensation (excess > 0) ---
    cond_mask = excess > 0.0
    if cond_mask.any():
        # Newton correction for latent heating feedback (1 iteration)
        # Physics: Condensing δ kg/kg heats air by ΔT = (Lv/cp)*δ (NOT π₀*δ!)
        # Because: Δθ = (Lv/cp)*δ/π₀, so ΔT = π₀*Δθ = (Lv/cp)*δ
        # This changes saturation: Δq_s = (dq_s/dT)*ΔT = (dq_s/dT)*(Lv/cp)*δ
        # So total change: δ = excess - Δq_s = excess - (dq_s/dT)*(Lv/cp)*δ
        # Solving: δ = excess / (1 + (Lv/cp)*(dq_s/dT))  [NO π₀!]
        L_cp  = cfg.LV / cfg.CP
        dqsdT = q_s * cfg.LV / (cfg.RV * T**2)            # dq_s/dT
        denom = 1.0 + L_cp * dqsdT                        # Corrected: no π₀ factor
        delta = excess / denom                             # kg/kg condensed

        np.add.at   # suppress "unused"
        s.qc    = np.where(cond_mask, s.qc + delta,    s.qc)
        s.qv    = np.where(cond_mask, s.qv - delta,    s.qv)
        # θ increases due to latent heating: Δθ = (Lv/cp) * δ / π₀
        s.theta = np.where(cond_mask,
                           s.theta + L_cp * delta / pi0,
                           s.theta)

    # --- evaporation of cloud water (excess < 0 and qc > 0) ---
    evap_mask = (excess < 0.0) & (s.qc > 0.0)
    if evap_mask.any():
        L_cp  = cfg.LV / cfg.CP
        dqsdT = q_s * cfg.LV / (cfg.RV * T**2)
        denom = 1.0 + L_cp * dqsdT                        # Corrected: no π₀ factor
        delta = np.minimum(-excess / denom, s.qc)         # can't evaporate more than exists

        s.qc    = np.where(evap_mask, s.qc - delta,    s.qc)
        s.qv    = np.where(evap_mask, s.qv + delta,    s.qv)
        s.theta = np.where(evap_mask,
                           s.theta - L_cp * delta / pi0,
                           s.theta)


# ── Turbulent entrainment ────────────────────────────────────────────────────

def apply_entrainment(s: State, g: Grid, dt: float) -> None:
    """
    Parameterize turbulent entrainment of environmental air into cloud edges.

    Physical process:
    - Turbulence at cloud boundaries mixes dry environmental air into cloud
    - Dilutes cloud water → evaporation → evaporative cooling
    - Creates wispy, turbulent cloud structure

    Method:
    - Entrainment rate ε (s^-1) applied at cloud boundaries
    - Gradient |∇qc| used to identify boundaries (weight 0-1)
    - Relax cloud properties toward horizontal-mean environment
    - Saturation adjustment handles resulting evaporation/cooling

    Parameters from config:
    - ENTRAINMENT_RATE: time rate (s^-1) (typical 0.001-0.01, ~100-1000s timescale)
    """
    if not cfg.ENABLE_ENTRAINMENT:
        return

    # Cloud threshold (0.1 g/kg)
    QC_THRESHOLD = 1e-4  # kg/kg

    # Identify cloudy regions
    cloud_mask = s.qc > QC_THRESHOLD
    if not cloud_mask.any():
        return  # No clouds, skip

    # Compute horizontal gradient of qc using centered differences
    dqc_dx = (np.roll(s.qc, -1, axis=0) - np.roll(s.qc, 1, axis=0)) / (2.0 * g.dx)
    dqc_dy = (np.roll(s.qc, -1, axis=1) - np.roll(s.qc, 1, axis=1)) / (2.0 * g.dy)
    grad_qc = np.sqrt(dqc_dx**2 + dqc_dy**2)

    # Entrainment rate: constant rate at cloud boundaries
    # Use gradient to identify boundaries (normalized to 0-1 range)
    # Typical cloud edge gradient: 1e-6 to 1e-5 kg/kg/m
    grad_threshold = 1e-6  # kg/kg/m, threshold for "cloud boundary"
    boundary_weight = np.minimum(grad_qc / grad_threshold, 1.0)  # dimensionless, 0-1

    # ENTRAINMENT_RATE is now in s^-1 (time rate, not length^-1)
    # Stronger entrainment at sharp boundaries, weaker in cloud core
    epsilon = cfg.ENTRAINMENT_RATE * boundary_weight  # s^-1

    # Optional: dynamic entrainment (scale with updraft velocity)
    # Physical motivation: stronger updrafts create more turbulence → more mixing
    # This creates temporal oscillations: active growth → strong entrainment → decay
    if cfg.ENABLE_DYNAMIC_ENTRAINMENT:
        # Vertical velocity at cell centers (interpolate from faces)
        w_centers = 0.5 * (s.w[:, :, :-1] + s.w[:, :, 1:])  # (nx, ny, nz)

        # Turbulence enhancement factor (1.0 baseline, increases with |w|)
        # Strong updrafts (w ~ 5-10 m/s) increase entrainment by 2-3x
        turbulence_factor = 1.0 + np.abs(w_centers) / cfg.DYNAMIC_ENTR_W_SCALE

        epsilon *= turbulence_factor  # s^-1, now varies with local dynamics

    # Environmental state = horizontal mean at each level
    qv_env = s.qv.mean(axis=(0, 1))[np.newaxis, np.newaxis, :]     # (1,1,nz)
    theta_env = s.theta.mean(axis=(0, 1))[np.newaxis, np.newaxis, :]

    # Relaxation toward environment (only in cloudy regions)
    # dφ/dt = -ε * (φ - φ_env)  →  φ(t+dt) = φ + dt * (-ε * (φ - φ_env))
    relax_factor = epsilon * dt
    relax_factor = np.minimum(relax_factor, 1.0)  # Limit to avoid overshoot

    # Apply entrainment mixing
    delta_qv = relax_factor * (qv_env - s.qv)
    delta_theta = relax_factor * (theta_env - s.theta)
    delta_qc = -relax_factor * s.qc  # Dilute cloud water

    s.qv = np.where(cloud_mask, s.qv + delta_qv, s.qv)
    s.theta = np.where(cloud_mask, s.theta + delta_theta, s.theta)
    s.qc = np.where(cloud_mask, s.qc + delta_qc, s.qc)

    # Ensure non-negative qc
    s.qc = np.maximum(s.qc, 0.0)


# ── Kessler collection / conversion ──────────────────────────────────────────

# Autoconversion and accretion parameters (now from config)
# _K1 and _QC0 moved to config.py as AUTOCONV_RATE and AUTOCONV_THRESHOLD
_K2    = 2.2     # accretion coefficient  (Kessler 1969, SI units)
_K_EVP = 2.0e-4  # legacy rain evaporation coefficient (RAIN_EVAP = "legacy")

# Seifert & Beheng (2001), mass densities L = ρq [kg/m³]
_SB_KC    = 9.44e9    # m³ kg⁻² s⁻¹, autoconversion kernel constant
_SB_KR    = 5.78      # m³ kg⁻¹ s⁻¹, accretion kernel constant
_SB_XSTAR = 2.6e-10   # kg, drop mass separating cloud from rain (r ≈ 40 µm)
_SB_NU    = 1.0       # cloud droplet size-distribution shape parameter
_SB_AU    = _SB_KC / (20.0 * _SB_XSTAR) * (_SB_NU + 2) * (_SB_NU + 4) / (_SB_NU + 1) ** 2


def _flags():
    """Integer switches for the kernels: (scheme 1 = sb2001 / 0 = kessler, evap 1 = kw78 / 0 = legacy)."""
    scheme = {"sb2001": 1, "kessler": 0}[cfg.RAIN_SCHEME]
    evap = {"kw78": 1, "legacy": 0}[cfg.RAIN_EVAP]
    return scheme, evap


def _rain_rates(qc, qr, rho):
    """Autoconversion + accretion rate [kg/kg/s] of the selected scheme."""
    if cfg.RAIN_SCHEME == "kessler":
        return (np.maximum(0.0, cfg.AUTOCONV_RATE * (qc - cfg.AUTOCONV_THRESHOLD))
                + _K2 * qc * qr ** 0.875)
    qt = qc + qr
    tau = np.clip(np.where(qt > 0, qr / np.maximum(qt, 1e-30), 0.0), 0.0, 0.9999)
    Lc, Lr = rho * qc, rho * qr
    xc = np.minimum(Lc / cfg.CLOUD_NC, _SB_XSTAR)                     # mean cloud droplet mass
    t68 = tau ** 0.68
    au = _SB_AU * Lc**2 * xc**2 * (1.0 + 600.0 * t68 * (1.0 - t68)**3 / (1.0 - tau)**2)
    ac = _SB_KR * Lc * Lr * (tau / (tau + 5e-4))**4
    return (au + ac) / rho


def _evap_rate(qr, qv, q_s, rho, p):
    """Rain evaporation rate [kg/kg/s] (≥ 0, only where sub-saturated)."""
    deficit = np.maximum(0.0, q_s - qv)
    if cfg.RAIN_EVAP == "legacy":
        return _K_EVP * deficit * qr ** 0.525
    rq = rho * 1e-3 * qr                                    # g/cm³
    C = 1.6 + 124.9 * rq ** 0.2046                          # ventilation
    return (deficit / q_s) * C * rq ** 0.525 / ((5.4e5 + 2.55e6 / (p * 1e-2 * q_s)) * rho * 1e-3)


def _lf_cp() -> float:
    """L_f/c_p when cloud ice exists (ICE_SCHEME "mixed"), else 0."""
    return cfg.LF / cfg.CP if thermodynamics._mixed() else 0.0


def warm_rain(s: State, g: Grid, dt: float) -> None:
    """
    Autoconversion, accretion, rain evaporation.  Updates qc, qr, theta in-place.
    """
    if _numba():
        theta0, pi0_1d, p = _profiles(s)
        s.theta, s.qv, s.qc, s.qr = (np.array(a, dtype=np.float64)
                                     for a in (s.theta, s.qv, s.qc, s.qr))
        scheme, evap = _flags()
        kernels.warm_rain(s.theta, s.qv, s.qc, s.qr, theta0, pi0_1d, p,
                          np.asarray(s.rho0, dtype=np.float64), cfg.EPS, dt,
                          scheme, evap, cfg.AUTOCONV_RATE, cfg.AUTOCONV_THRESHOLD, _K2, _K_EVP,
                          cfg.CLOUD_NC, _SB_AU, _SB_KR, _SB_XSTAR, cfg.LV / cfg.CP,
                          _lf_cp(), cfg.T_ICE_0, cfg.T_ICE_00)
        return
    pi0 = s._pi0()
    theta_abs = s.theta + s.theta0[np.newaxis, np.newaxis, :]
    T   = theta_abs * pi0
    q_s = _q_sat(T, pi0)

    rho = s.rho0[np.newaxis, np.newaxis, :]
    p = cfg.P00 * pi0 ** (cfg.CP / cfg.RD)

    # Autoconversion + accretion: cloud → rain
    conv = np.minimum(_rain_rates(s.qc, s.qr, rho) * dt, s.qc)
    s.qc -= conv
    s.qr += conv
    if _lf_cp() > 0.0:                                       # rain is liquid: the ice part melts
        s.theta -= _lf_cp() * (1.0 - thermodynamics.liquid_fraction(T)[0]) * conv / pi0

    # Rain evaporation (only where sub-saturated; never beyond saturation)
    evap = _evap_rate(s.qr, s.qv, q_s, rho, p) * dt
    if cfg.RAIN_EVAP == "kw78":
        evap = np.minimum(evap, np.maximum(0.0, q_s - s.qv))
    evap = np.minimum(evap, s.qr)

    s.qr   -= evap
    s.qv   += evap
    s.theta -= (cfg.LV / cfg.CP) * evap / pi0


# ── Rain sedimentation ────────────────────────────────────────────────────────

_A_VT = 36.34   # terminal velocity coefficient
_B_VT = 0.1364  # exponent


def sedimentation(s: State, g: Grid, dt: float) -> None:
    """
    Implicit upwind sedimentation using terminal velocity only.

    Fixed algorithm (top-down sweep):
    - Terminal velocity (Klemp & Wilhelmson 1978, Kessler form):
      V_t = A_VT * (rho*qr [g/cm³])^B_VT * sqrt(rho_sfc/rho)
    - Rain is already advected by resolved flow (u, v, w), so we only add sedimentation
    - Solve implicitly from top (k=nz-1) down to surface (k=0) for mass conservation
    - Density weighting: use rho0*qr to conserve rain mass
    - Surface precip flux: rho0[0] * V_t[0] * qr[0] (kg/m²/s) → mm/hr conversion

    Algorithm:
      for k from nz-1 down to 0:
          F_in[k] = F_out[k+1] if k < nz-1 else 0  (flux from above)
          qr[k] = (qr[k] + dt/dz[k] * F_in[k]/rho0[k]) / (1 + V_t[k]*dt/dz[k])
          F_out[k] = V_t[k] * rho0[k] * qr[k]  (mass flux leaving this cell)
    """
    kb = None if g.terrain is None else g.terrain.k_sfc    # terrain: rain reaches the ground at k_sfc
    if kb is not None:                                      # fluid thickness of the ground cell
        hcg = np.take_along_axis(g.terrain.hc, kb[:, :, None], axis=2)[:, :, 0]
        dzb = hcg * g.dz_face[kb]
    if _numba():
        s.qr = np.array(s.qr, dtype=np.float64)
        args = (s.qr, np.asarray(s.rho0, dtype=np.float64), np.asarray(g.dz_face, dtype=np.float64),
                dt, _A_VT, _B_VT)
        sfc = (kernels.sedimentation(*args) if kb is None
               else kernels.sedimentation_kb(*args, np.ascontiguousarray(kb, dtype=np.int64),
                                             np.ascontiguousarray(dzb, dtype=np.float64)))
        s.surface_precip_flux = sfc * 3600.0
        return

    rho0 = s.rho0[np.newaxis, np.newaxis, :]  # (1,1,nz)
    dz_f = g.dz_face[np.newaxis, np.newaxis, :]  # (1,1,nz)

    # Ensure non-negative qr before computing terminal velocity
    s.qr = np.maximum(s.qr, 0.0)

    # Terminal velocity (Klemp & Wilhelmson 1978):
    # V_t = A_VT * (rho*qr [g/cm³])^B_VT * sqrt(rho_sfc/rho)
    # Unit conversion: rho0 [kg/m³] * qr [kg/kg] → [kg/m³] → [g/cm³] (divide by 1000)
    rho_qr_SI = rho0 * s.qr  # kg/m³
    rho_qr_cgs = rho_qr_SI / 1000.0  # g/cm³

    # Density correction factor: sqrt(rho_sfc / rho)
    rho_sfc = s.rho0[0]
    density_correction = np.sqrt(rho_sfc / rho0)  # (1,1,nz)

    V_t = _A_VT * (rho_qr_cgs ** _B_VT) * density_correction  # (nx,ny,nz) [m/s]

    # Top-down sweep: solve implicit sedimentation layer by layer
    # F_out[k] = mass flux leaving cell k (downward, kg/m²/s)
    F_out = np.zeros_like(s.qr)  # (nx, ny, nz)

    active = None
    for k in range(g.nz - 1, -1, -1):  # k = nz-1, nz-2, ..., 1, 0
        if kb is not None:
            active = k >= kb                       # (nx, ny): this level is above the ground
            if not active.any():
                break
        # Flux entering from above
        if k == g.nz - 1:
            F_in_k = 0.0  # Top boundary: no flux from above
        else:
            F_in_k = F_out[:, :, k + 1]  # Flux from cell above (k+1)

        # Implicit update: qr[k] = (qr[k] + dt/dz * F_in/rho0) / (1 + V_t*dt/dz)
        # This conserves mass: d(rho0*qr)/dt = -(F_out - F_in)/dz
        dz_k = dz_f[0, 0, k]
        rho0_k = rho0[0, 0, k]
        V_t_k = V_t[:, :, k]

        # Update qr (implicit); terrain: the ground cell's fluid thickness (partial cells)
        if active is not None:
            dz_k = np.where(kb == k, dzb, dz_k)
        denominator = 1.0 + V_t_k * dt / dz_k
        q_new = np.maximum((s.qr[:, :, k] + (dt / dz_k) * (F_in_k / rho0_k)) / denominator, 0.0)
        if active is None:
            s.qr[:, :, k] = q_new
        else:
            s.qr[:, :, k] = np.where(active, q_new, s.qr[:, :, k])

        # Compute outgoing flux (mass flux, kg/m²/s)
        F_out[:, :, k] = V_t_k * rho0_k * s.qr[:, :, k]
        if active is not None:
            F_out[:, :, k] = np.where(active, F_out[:, :, k], 0.0)

    # Surface precipitation flux: F_out[k=0] is rain leaving the domain
    # Convert from kg/m²/s to mm/hr: 1 kg/m²/s = 3600 mm/hr (since 1 kg/m² water = 1 mm depth)
    if kb is None:
        s.surface_precip_flux = F_out[:, :, 0] * 3600.0  # (nx, ny) in mm/hr
    else:                                                  # flux out of each column's ground cell
        s.surface_precip_flux = np.take_along_axis(F_out, kb[:, :, None], axis=2)[:, :, 0] * 3600.0

    # Final safety check
    s.qr = np.maximum(s.qr, 0.0)


# ── Main microphysics driver ──────────────────────────────────────────────────

def apply_microphysics(s: State, g: Grid, dt: float) -> None:
    """
    Apply microphysical processes in sequence:
    1. Saturation adjustment (condensation/evaporation)
    2. Entrainment (turbulent mixing at cloud edges) - OPTIONAL
    3. Warm rain (autoconversion, accretion, rain evaporation)
    4. Sedimentation (rain fallout)
    """
    saturation_adjustment(s, dt)

    # Apply entrainment if enabled (creates wispy cloud structure)
    if cfg.ENABLE_ENTRAINMENT:
        apply_entrainment(s, g, dt)
        # Re-run saturation adjustment after entrainment
        # (diluted qc may evaporate, creating cooling)
        saturation_adjustment(s, dt)

    warm_rain(s, g, dt)
    sedimentation(s, g, dt)

    # Enforce non-negative mixing ratios
    s.qv = np.maximum(s.qv, 0.0)
    s.qc = np.maximum(s.qc, 0.0)
    s.qr = np.maximum(s.qr, 0.0)
