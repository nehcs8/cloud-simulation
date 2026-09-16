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
from grid import Grid
from state import State


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
    """
    pi0 = s._pi0()                                         # (1,1,nz)
    theta_abs = s.theta + s.theta0[np.newaxis, np.newaxis, :]
    T   = theta_abs * pi0                                  # (nx,ny,nz)
    q_s = _q_sat(T, pi0)

    excess = s.qv - q_s                                    # positive → condense

    # --- condensation (excess > 0) ---
    cond_mask = excess > 0.0
    if cond_mask.any():
        # Newton correction for latent heating feedback (1 iteration)
        L_cp  = cfg.LV / cfg.CP
        dqsdT = q_s * cfg.LV / (cfg.RV * T**2)            # dq_s/dT
        denom = 1.0 + L_cp * dqsdT
        delta = excess / denom                             # kg/kg condensed

        np.add.at   # suppress "unused"
        s.qc    = np.where(cond_mask, s.qc + delta,    s.qc)
        s.qv    = np.where(cond_mask, s.qv - delta,    s.qv)
        # θ increases due to latent heating: Δθ = (Lv/cp) * Δqv / π₀
        s.theta = np.where(cond_mask,
                           s.theta + L_cp * delta / pi0,
                           s.theta)

    # --- evaporation of cloud water (excess < 0 and qc > 0) ---
    evap_mask = (excess < 0.0) & (s.qc > 0.0)
    if evap_mask.any():
        L_cp  = cfg.LV / cfg.CP
        dqsdT = q_s * cfg.LV / (cfg.RV * T**2)
        denom = 1.0 + L_cp * dqsdT
        delta = np.minimum(-excess / denom, s.qc)         # can't evaporate more than exists

        s.qc    = np.where(evap_mask, s.qc - delta,    s.qc)
        s.qv    = np.where(evap_mask, s.qv + delta,    s.qv)
        s.theta = np.where(evap_mask,
                           s.theta - L_cp * delta / pi0,
                           s.theta)


# ── Kessler collection / conversion ──────────────────────────────────────────

_K1    = 1e-3    # s^-1,  autoconversion rate
_QC0   = 2e-4    # kg/kg, autoconversion threshold (lower → more rain produced)
_K2    = 2.2     # accretion coefficient  (Kessler 1969, SI units)
_K_EVP = 2.0e-4  # rain evaporation coefficient (lower → rain penetrates deeper)


def warm_rain(s: State, g: Grid, dt: float) -> None:
    """
    Autoconversion, accretion, rain evaporation.  Updates qc, qr, theta in-place.
    """
    pi0 = s._pi0()
    theta_abs = s.theta + s.theta0[np.newaxis, np.newaxis, :]
    T   = theta_abs * pi0
    q_s = _q_sat(T, pi0)

    # Autoconversion: cloud → rain
    auto = np.maximum(0.0, _K1 * (s.qc - _QC0)) * dt
    auto = np.minimum(auto, s.qc)

    # Accretion: rain sweeps up cloud water
    accr = _K2 * s.qc * s.qr ** 0.875 * dt
    accr = np.minimum(accr, s.qc)

    s.qc -= (auto + accr)
    s.qr += (auto + accr)

    # Rain evaporation (only where sub-saturated and rain exists)
    deficit = np.maximum(0.0, q_s - s.qv)
    evap    = _K_EVP * deficit * s.qr ** 0.525 * dt
    evap    = np.minimum(evap, s.qr)

    s.qr   -= evap
    s.qv   += evap
    s.theta -= (cfg.LV / cfg.CP) * evap / pi0


# ── Rain sedimentation ────────────────────────────────────────────────────────

_A_VT = 36.34   # terminal velocity coefficient
_B_VT = 0.1364  # exponent


def sedimentation(s: State, g: Grid, dt: float) -> None:
    """
    Implicit upstream sedimentation of rain.
    V_t = A_VT * (rho0 * qr)^B_VT  (fall speed, positive downward).
    """
    rho0 = s.rho0[np.newaxis, np.newaxis, :]           # (1,1,nz)
    V_t  = _A_VT * (rho0 * np.maximum(s.qr, 0.0)) ** _B_VT   # (nx,ny,nz)

    dz_f = g.dz_face[np.newaxis, np.newaxis, :]        # (1,1,nz)

    # Upstream (downward) flux: flux at top face of cell k  = V_t[k] * qr[k]
    flux_in  = np.zeros_like(s.qr)
    flux_out = V_t * s.qr

    # Flux entering from above (k-1) — shift down
    flux_in[:, :, 1:] = flux_out[:, :, :-1]
    # Bottom boundary: rain exits domain
    flux_in[:, :, 0] = 0.0

    # Implicit factor for stability
    implicit = 1.0 + V_t * dt / dz_f
    s.qr = (s.qr + dt / dz_f * flux_in) / implicit
    s.qr = np.maximum(s.qr, 0.0)


# ── Main microphysics driver ──────────────────────────────────────────────────

def apply_microphysics(s: State, g: Grid, dt: float) -> None:
    saturation_adjustment(s, dt)
    warm_rain(s, g, dt)
    sedimentation(s, g, dt)
    # Enforce non-negative mixing ratios
    s.qv = np.maximum(s.qv, 0.0)
    s.qc = np.maximum(s.qc, 0.0)
    s.qr = np.maximum(s.qr, 0.0)
