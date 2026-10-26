"""
Kessler warm-rain bulk microphysics.

Processes (applied in order each timestep):
  1. Saturation adjustment  – condensation / evaporation of cloud water
  2. Autoconversion         – cloud water → rain  (Kessler 1969)
  3. Accretion              – rain collecting cloud droplets
  4. Rain evaporation       – sub-saturated air evaporates rain
  5. Sedimentation          – rain falls at terminal velocity
ICE_SCHEME "mixed"/"graupel": cloud condensate is split into liquid/ice by temperature
(thermodynamics.py); "graupel" adds precipitating ice q_g (ice_graupel) between 4 and 5.
"""

import math
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


# ── Process budget ───────────────────────────────────────────────────────────
# Mass converted by each process, accumulated over the run in s.micro_acc (nx, nz, n): Σ_y Δq [kg/kg] per (x, level);
# the two surface entries hold Σ_y (precip · dt) [kg/m²] at level 0. budget_kg() turns it into kg per level.
# Cloud condensation/evaporation is not listed: with θ_l/q_t it happens implicitly at every diagnosis.
PROCESSES = ('cloud_to_rain', 'rain_evap', 'ice_autoconv', 'ice_collect', 'riming', 'shedding',
             'rain_freeze', 'melting', 'graupel_subl', 'rain_sfc', 'graupel_sfc')
_SFC = (PROCESSES.index('rain_sfc'), PROCESSES.index('graupel_sfc'))


def _acc(s: State) -> np.ndarray:
    a = getattr(s, 'micro_acc', None)
    nx, nz = s.qc.shape[0], s.qc.shape[2]
    if a is None or a.shape != (nx, nz, len(PROCESSES)):
        s.micro_acc = a = np.zeros((nx, nz, len(PROCESSES)))
    return a


def budget_kg(s: State, g: Grid) -> np.ndarray:
    """Cumulative mass per process and level [kg], shape (nz, len(PROCESSES)) (surface entries at level 0)."""
    a = _acc(s).sum(axis=0)
    out = a * (s.rho0 * g.dx * g.dy * g.dz_face)[:, None]
    out[:, list(_SFC)] = a[:, list(_SFC)] * g.dx * g.dy
    return out


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


def _two_mom() -> bool:
    return getattr(cfg, 'RAIN_MOMENTS', 1) == 2


# ── Two-moment rain (RAIN_MOMENTS = 2) ───────────────────────────────────────
# Exponential drop sizes from (q_r, n_r): mean mass x = q_r/n_r (clipped to RAIN_XMIN … RAIN_XMAX), λ = (π ρ_w / x)^⅓.
# Evaporation uses the Lin-type ventilated formula with V = 842 D^0.8 (as for graupel); sedimentation uses the
# Atlas et al. (1973) V(D) moments. The numba twins are kernels.rain_2m / sediment_2m.
_X_SHED = 5.2e-7                                     # kg, mass of a 1 mm drop (shed meltwater)
_RAIN_A, _RAIN_B = 842.0, 0.8


def _c_vent_rain():
    return 0.31 * _SC ** (1.0 / 3.0) * math.gamma(0.5 * (_RAIN_B + 5.0)) * np.sqrt(_RAIN_A / _NU)


def _rain_lambda(q, n):
    x = np.clip(q / np.where(n > 0, n, 1.0), cfg.RAIN_XMIN, cfg.RAIN_XMAX)
    return (np.pi * 1000.0 / x) ** (1.0 / 3.0), x


def _clip_nr(q, n):
    return np.where(q <= 1e-15, 0.0, np.clip(n, q / cfg.RAIN_XMAX, q / cfg.RAIN_XMIN))


def rain_2m(s: State, g: Grid, dt: float) -> None:
    """Two-moment warm rain: SB2001 autoconversion (adds drops of mass x*) and accretion, SB2006 self-collection
    and breakup (mean diameter → RAIN_DEQ), evaporation of the exponential (q_r, n_r) distribution with the
    number lost in proportion. Updates θ′, q_v, q_c, q_r, n_r in place."""
    lf = _lf_cp(); liq = int(thermodynamics._graupel())
    if _numba():
        theta0, pi0_1d, p = _profiles(s)
        s.theta, s.qv, s.qc, s.qr, s.nr = (np.array(a, dtype=np.float64)
                                           for a in (s.theta, s.qv, s.qc, s.qr, s.nr))
        kernels.rain_2m(s.theta, s.qv, s.qc, s.qr, s.nr, theta0, pi0_1d, p,
                        np.asarray(s.rho0, dtype=np.float64), cfg.EPS, dt, cfg.CLOUD_NC, _SB_AU, _SB_KR,
                        _SB_XSTAR, cfg.LV / cfg.CP, lf, cfg.T_ICE_0, cfg.T_ICE_00, liq, cfg.RAIN_KRR,
                        cfg.RAIN_DEQ, cfg.RAIN_XMIN, cfg.RAIN_XMAX, _c_vent_rain(), _KA, _PSI, cfg.RV, cfg.CP,
                        _acc(s))
        return
    pi0 = s._pi0()
    p = cfg.P00 * pi0 ** (cfg.CP / cfg.RD)
    rho = s.rho0[np.newaxis, np.newaxis, :]
    rfac = np.sqrt(s.rho0[0] / rho)
    T = (s.theta + s.theta0[np.newaxis, np.newaxis, :]) * pi0
    q_s = _q_sat(T, pi0)
    L_cp = cfg.LV / cfg.CP; lv = cfg.LV
    a = thermodynamics.liquid_fraction(T)[0] if lf > 0.0 else 1.0
    c, r, n = s.qc, s.qr, s.nr
    cl = a * c if liq else c
    qt = cl + r
    tau = np.clip(np.where(qt > 0, r / np.where(qt > 0, qt, 1.0), 0.0), 0.0, 0.9999)
    Lc, Lr = rho * cl, rho * r
    xc = np.minimum(Lc / cfg.CLOUD_NC, _SB_XSTAR)
    t68 = tau ** 0.68
    au = _SB_AU * Lc * Lc * xc * xc * (1.0 + 600.0 * t68 * (1.0 - t68) ** 3 / (1.0 - tau) ** 2) / rho
    ac = _SB_KR * Lc * Lr * (tau / (tau + 5e-4)) ** 4 / rho
    conv = np.minimum((au + ac) * dt, cl)
    d_au = np.where(conv > 0, conv * au / np.where(au + ac > 0, au + ac, 1.0), 0.0)
    c = c - conv; r = r + conv; n = n + d_au / _SB_XSTAR
    if lf > 0.0:
        s.theta = s.theta - lf * (1.0 - a) * conv / pi0
    act = (r > 1e-12) & (n > 0)
    lam, x = _rain_lambda(r, n)
    dm = (6.0 * x / (np.pi * 1000.0)) ** (1.0 / 3.0)
    sc = cfg.RAIN_KRR * n * rho * r * rfac
    phi = np.where(dm < 0.35e-3, 0.0, np.where(dm <= cfg.RAIN_DEQ, 1.0 + 1000.0 * (dm - cfg.RAIN_DEQ),
                                               2.0 * np.exp(2300.0 * (dm - cfg.RAIN_DEQ)) - 1.0))
    n_sc = np.maximum(n + (phi - 1.0) * sc * dt / (1.0 + sc * dt / np.where(n > 0, n, 1.0)),
                      r / cfg.RAIN_XMAX)
    n = np.where(act, n_sc, n)
    deficit = q_s - s.qv
    ev = act & (deficit > 0)
    lam, x = _rain_lambda(r, n)
    abw = lv * lv / (_KA * cfg.RV * T * T) + 1.0 / (rho * q_s * _PSI)
    vent = 0.78 / lam**2 + _c_vent_rain() * np.sqrt(rfac) * lam ** (-0.5 * (_RAIN_B + 5.0))
    e = 2.0 * np.pi * rho * n * lam * (deficit / q_s) * vent / (rho * abw)
    evap = np.where(ev, np.minimum(np.minimum(e * dt, deficit / (1.0 + L_cp * q_s * lv / (cfg.RV * T * T))), r), 0.0)
    n = n - np.where(ev, n * evap / np.where(r > 0, r, 1.0), 0.0)
    r = r - evap
    s.qc, s.qr, s.nr = c, r, _clip_nr(r, n)
    s.qv = s.qv + evap
    s.theta = s.theta - L_cp * evap / pi0
    acc = _acc(s)
    acc[:, :, 0] += conv.sum(axis=1)
    acc[:, :, 1] += evap.sum(axis=1)


def _sediment_2m(s: State, g: Grid, dt: float):
    """Two-moment rain sedimentation (Atlas V(D) moments); returns surface rain [mm/h]. Updates s.qr, s.nr."""
    nx, ny, nz = s.qr.shape
    if g.terrain is None:
        kb = np.zeros((nx, ny), dtype=np.int64)
        dzb = np.full((nx, ny), float(g.dz_face[0]))
    else:
        kb = g.terrain.k_sfc.astype(np.int64)
        dzb = np.take_along_axis(g.terrain.hc, kb[:, :, None], axis=2)[:, :, 0] * g.dz_face[kb]
    rho0 = np.asarray(s.rho0, dtype=np.float64)
    if _numba():
        s.qr, s.nr = np.array(s.qr, dtype=np.float64), np.array(s.nr, dtype=np.float64)
        sfc = kernels.sediment_2m(s.qr, s.nr, rho0, np.asarray(g.dz_face, dtype=np.float64), dt,
                                  np.ascontiguousarray(kb), np.ascontiguousarray(dzb, dtype=np.float64),
                                  cfg.RAIN_XMIN, cfg.RAIN_XMAX)
        return sfc * 3600.0
    q, n = np.maximum(s.qr, 0.0), np.maximum(s.nr, 0.0)
    Fq = np.zeros((nx, ny)); Fn = np.zeros((nx, ny))
    for k in range(nz - 1, -1, -1):
        on = k >= kb
        if not on.any():
            break
        qk, nk = q[:, :, k], n[:, :, k]
        live = (qk > 1e-12) & (nk > 0)
        lam, _ = _rain_lambda(qk, nk)
        f = lam / (lam + 600.0)
        rf = np.sqrt(rho0[0] / rho0[k])
        vq = np.where(live, np.maximum(9.65 - 10.3 * f**4, 0.1) * rf, 0.0)
        vn = np.where(live, np.maximum(9.65 - 10.3 * f, 0.1) * rf, 0.0)
        dzk = np.where(kb == k, dzb, g.dz_face[k])
        qn = np.maximum((qk + (dt / dzk) * (Fq / rho0[k])) / (1.0 + vq * dt / dzk), 0.0)
        nn = np.maximum((nk + (dt / dzk) * (Fn / rho0[k])) / (1.0 + vn * dt / dzk), 0.0)
        Fq = np.where(on, vq * rho0[k] * qn, Fq)
        Fn = np.where(on, vn * rho0[k] * nn, Fn)
        q[:, :, k] = np.where(on, qn, qk)
        n[:, :, k] = np.where(on, _clip_nr(qn, nn), nk)
    s.qr, s.nr = q, n
    return Fq * 3600.0


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
                          _lf_cp(), cfg.T_ICE_0, cfg.T_ICE_00, int(thermodynamics._graupel()), _acc(s))
        return
    pi0 = s._pi0()
    theta_abs = s.theta + s.theta0[np.newaxis, np.newaxis, :]
    T   = theta_abs * pi0
    q_s = _q_sat(T, pi0)

    rho = s.rho0[np.newaxis, np.newaxis, :]
    p = cfg.P00 * pi0 ** (cfg.CP / cfg.RD)

    # Autoconversion + accretion: cloud → rain
    alpha = thermodynamics.liquid_fraction(T)[0] if _lf_cp() > 0.0 else 1.0
    cl = alpha * s.qc if thermodynamics._graupel() else s.qc   # "graupel": rain from the liquid part only
    conv = np.minimum(_rain_rates(cl, s.qr, rho) * dt, cl)
    s.qc -= conv
    s.qr += conv
    if _lf_cp() > 0.0:            # q_c keeps its split α(T): removing Δ relabels (1 − α)Δ liquid → ice
        s.theta -= _lf_cp() * (1.0 - alpha) * conv / pi0

    # Rain evaporation (only where sub-saturated; never beyond saturation)
    evap = _evap_rate(s.qr, s.qv, q_s, rho, p) * dt
    if cfg.RAIN_EVAP == "kw78":
        evap = np.minimum(evap, np.maximum(0.0, q_s - s.qv))
    evap = np.minimum(evap, s.qr)

    s.qr   -= evap
    s.qv   += evap
    acc = _acc(s)
    acc[:, :, 0] += conv.sum(axis=1)
    acc[:, :, 1] += evap.sum(axis=1)
    s.theta -= (cfg.LV / cfg.CP) * evap / pi0


# ── Rain sedimentation ────────────────────────────────────────────────────────

_A_VT = 36.34   # terminal velocity coefficient
_B_VT = 0.1364  # exponent


def _sediment(q: np.ndarray, s: State, g: Grid, dt: float, a_vt: float, b_vt: float):
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

    Generic for any precipitation species with V_t = a_vt (ρq [g/cm³])^b_vt √(ρ_sfc/ρ):
    returns (q after sedimentation, surface flux [mm/h]).
    """
    kb = None if g.terrain is None else g.terrain.k_sfc    # terrain: rain reaches the ground at k_sfc
    if kb is not None:                                      # fluid thickness of the ground cell
        hcg = np.take_along_axis(g.terrain.hc, kb[:, :, None], axis=2)[:, :, 0]
        dzb = hcg * g.dz_face[kb]
    if _numba():
        q = np.array(q, dtype=np.float64)
        args = (q, np.asarray(s.rho0, dtype=np.float64), np.asarray(g.dz_face, dtype=np.float64),
                dt, a_vt, b_vt)
        sfc = (kernels.sedimentation(*args) if kb is None
               else kernels.sedimentation_kb(*args, np.ascontiguousarray(kb, dtype=np.int64),
                                             np.ascontiguousarray(dzb, dtype=np.float64)))
        return q, sfc * 3600.0

    rho0 = s.rho0[np.newaxis, np.newaxis, :]  # (1,1,nz)
    dz_f = g.dz_face[np.newaxis, np.newaxis, :]  # (1,1,nz)

    # Ensure non-negative qr before computing terminal velocity
    q = np.maximum(q, 0.0)

    # Terminal velocity (Klemp & Wilhelmson 1978):
    # V_t = A_VT * (rho*qr [g/cm³])^B_VT * sqrt(rho_sfc/rho)
    # Unit conversion: rho0 [kg/m³] * qr [kg/kg] → [kg/m³] → [g/cm³] (divide by 1000)
    rho_qr_SI = rho0 * q  # kg/m³
    rho_qr_cgs = rho_qr_SI / 1000.0  # g/cm³

    # Density correction factor: sqrt(rho_sfc / rho)
    rho_sfc = s.rho0[0]
    density_correction = np.sqrt(rho_sfc / rho0)  # (1,1,nz)

    V_t = a_vt * (rho_qr_cgs ** b_vt) * density_correction  # (nx,ny,nz) [m/s]

    # Top-down sweep: solve implicit sedimentation layer by layer
    # F_out[k] = mass flux leaving cell k (downward, kg/m²/s)
    F_out = np.zeros_like(q)  # (nx, ny, nz)

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
        q_new = np.maximum((q[:, :, k] + (dt / dz_k) * (F_in_k / rho0_k)) / denominator, 0.0)
        if active is None:
            q[:, :, k] = q_new
        else:
            q[:, :, k] = np.where(active, q_new, q[:, :, k])

        # Compute outgoing flux (mass flux, kg/m²/s)
        F_out[:, :, k] = V_t_k * rho0_k * q[:, :, k]
        if active is not None:
            F_out[:, :, k] = np.where(active, F_out[:, :, k], 0.0)

    # Surface precipitation flux: F_out[k=0] is rain leaving the domain
    # Convert from kg/m²/s to mm/hr: 1 kg/m²/s = 3600 mm/hr (since 1 kg/m² water = 1 mm depth)
    if kb is None:
        sfc = F_out[:, :, 0] * 3600.0  # (nx, ny) in mm/hr
    else:                                                  # flux out of each column's ground cell
        sfc = np.take_along_axis(F_out, kb[:, :, None], axis=2)[:, :, 0] * 3600.0

    # Final safety check
    return np.maximum(q, 0.0), sfc


def sedimentation(s: State, g: Grid, dt: float) -> None:
    """Rain (and graupel, ICE_SCHEME "graupel") fall-out; s.surface_precip_flux = their sum [mm/h]."""
    acc = _acc(s)
    if _two_mom():
        sfc = _sediment_2m(s, g, dt)
    else:
        s.qr, sfc = _sediment(s.qr, s, g, dt, _A_VT, _B_VT)
    acc[:, 0, _SFC[0]] += sfc.sum(axis=1) * dt / 3600.0          # mm/h → kg/m² per step
    if thermodynamics._graupel():
        a_vt, b_vt = _graupel_vt()
        s.qg, sfc_g = _sediment(s.qg, s, g, dt, a_vt, b_vt)
        acc[:, 0, _SFC[1]] += sfc_g.sum(axis=1) * dt / 3600.0
        sfc = sfc + sfc_g
    s.surface_precip_flux = sfc


# ── Graupel (ICE_SCHEME = "graupel") ─────────────────────────────────────────
# One category of precipitating ice, inverse-exponential sizes n(D) = n0 e^(−λD), m = π/6 ρ_g D³,
# V = a D^b √(ρ_sfc/ρ): λ = (π ρ_g n0 / (ρ q_g))^¼, mass-weighted V = a Γ(4+b)/6 λ^(−b) √(ρ_sfc/ρ).
# Rates after Lin et al. (1983) / Rutledge & Hobbs (1984); cloud ice and liquid are the α(T) split of q_c.
_KA, _NU, _SC, _PSI = 2.43e-2, 1.5e-5, 0.6, 2.26e-5   # air conductivity, viscosity, Schmidt no., vapour diffusivity


def _graupel_vt():
    """(a_vt, b_vt) of the sedimentation power law for graupel, V = a_vt (ρq [g/cm³])^b_vt √(ρ_sfc/ρ)."""
    b = cfg.GR_B
    c = cfg.GR_A * math.gamma(4.0 + b) / 6.0 * (np.pi * cfg.GR_RHO * cfg.GR_N0) ** (-b / 4.0)
    return c * 1000.0 ** (b / 4.0), b / 4.0


def _graupel_consts():
    b = cfg.GR_B
    c_coll = np.pi / 4.0 * cfg.GR_N0 * cfg.GR_A * math.gamma(3.0 + b)
    c_vent = 0.31 * _SC ** (1.0 / 3.0) * math.gamma(0.5 * (b + 5.0)) * np.sqrt(cfg.GR_A / _NU)
    return c_coll, c_vent


def ice_graupel(s: State, g: Grid, dt: float) -> None:
    """
    Graupel processes (in place on θ′, q_v, q_c, q_r, q_g):
      cloud ice → graupel   autoconversion 1e-3·e^(0.025 T_c)·(q_i − ICE_AUTO_QI0) + collection (GR_E_ICE)
      cloud liquid → graupel  riming (T < 0 °C); above 0 °C the collected water is shed as rain
      rain → graupel         collection by graupel + Bigg (1953) freezing (T < 0 °C)
      graupel → rain         melting (T > 0 °C; never cools the cell below 0 °C in one step)
      graupel → vapour       sublimation where q_v < q_si
    Heating Δθ = [L_v (−Δq_v) + L_f Δ(ice)] / (c_p π₀), ice = (1 − α) q_c + q_g at the cell's α(T),
    so the diagnostic liquid/ice split of q_c stays energy-consistent.
    """
    lv_cp, lf_cp = cfg.LV / cfg.CP, cfg.LF / cfg.CP
    c_coll, c_vent = _graupel_consts()
    if _numba():
        theta0, pi0_1d, p = _profiles(s)
        s.theta, s.qv, s.qc, s.qr, s.qg, s.nr = (np.array(a, dtype=np.float64)
                                                 for a in (s.theta, s.qv, s.qc, s.qr, s.qg, s.nr))
        kernels.ice_graupel(s.theta, s.qv, s.qc, s.qr, s.qg, theta0, pi0_1d, p,
                            np.asarray(s.rho0, dtype=np.float64), cfg.EPS, dt, lv_cp, lf_cp,
                            cfg.T_ICE_0, cfg.T_ICE_00, cfg.GR_N0, cfg.GR_RHO, cfg.GR_A, cfg.GR_B,
                            cfg.GR_E_ICE, cfg.ICE_AUTO_QI0, c_coll, c_vent, _KA, _NU, _SC, _PSI,
                            cfg.RV, cfg.CP, _acc(s), s.nr, int(_two_mom()), _X_SHED)
        return
    pi0 = s._pi0()
    p = cfg.P00 * pi0 ** (cfg.CP / cfg.RD)
    rho = s.rho0[np.newaxis, np.newaxis, :]
    rfac = np.sqrt(s.rho0[0] / rho)
    T = (s.theta + s.theta0[np.newaxis, np.newaxis, :]) * pi0
    Tc = T - 273.15
    cold = Tc < 0.0
    a, _ = thermodynamics.liquid_fraction(T)
    c, r, gq, qv = s.qc, s.qr, s.qg, s.qv
    ql = a * c; qi = c - ql
    ls = cfg.LV + cfg.LF

    aut = np.where(cold & (qi > cfg.ICE_AUTO_QI0),
                   1e-3 * np.exp(0.025 * Tc) * (qi - cfg.ICE_AUTO_QI0), 0.0)
    has_r = cold & (r > 1e-12)
    lam_r = (np.pi * 1000.0 * 8.0e6 / (rho * np.where(has_r, r, 1.0))) ** 0.25
    gfr = np.where(has_r, 20.0 * np.pi**2 * 100.0 * 8.0e6 * (1000.0 / rho)
                   * (np.exp(-0.66 * Tc) - 1.0) / lam_r**7, 0.0)
    has_g = gq > 1e-12
    lam = (np.pi * cfg.GR_RHO * cfg.GR_N0 / (rho * np.where(has_g, gq, 1.0))) ** 0.25
    coll = np.where(has_g, c_coll * lam ** (-(3.0 + cfg.GR_B)) * rfac, 0.0)
    gacw = coll * ql
    gaci = np.where(cold, cfg.GR_E_ICE * coll * qi, 0.0)
    gacr = np.where(cold, coll * r, 0.0)
    vent = 0.78 / lam**2 + c_vent * np.sqrt(rfac) * lam ** (-0.5 * (cfg.GR_B + 5.0))
    melt = np.where(has_g & (Tc > 0.0), 2.0 * np.pi * cfg.GR_N0 * _KA * Tc * vent / (rho * cfg.LF), 0.0)
    e_i = 611.15 * np.exp(22.452 * Tc / (Tc + 272.55))
    qsi = cfg.EPS * e_i / np.maximum(p - e_i, 1.0)
    subl = has_g & (qv < qsi)
    abi = ls * ls / (_KA * cfg.RV * T * T) + 1.0 / (rho * qsi * _PSI)
    sub = np.where(subl, 2.0 * np.pi * cfg.GR_N0 * (1.0 - qv / qsi) * vent / (rho * abi), 0.0)
    sub = np.where(subl, np.minimum(sub * dt, (qsi - qv) / (1.0 + ls / cfg.CP * qsi * ls / (cfg.RV * T * T))) / dt, 0.0)

    d_i = np.minimum((aut + gaci) * dt, qi)
    d_l = np.minimum(gacw * dt, ql)
    d_r = np.minimum((gacr + gfr) * dt, r)
    d_m = np.minimum(np.minimum(melt * dt, gq), np.maximum(Tc, 0.0) / lf_cp)
    d_s = np.minimum(sub * dt, gq - d_m)
    to_g = d_i + d_r - d_m - d_s + np.where(cold, d_l, 0.0)
    to_r = d_m - d_r + np.where(cold, 0.0, d_l)
    if _two_mom():
        n = s.nr - np.where(d_r > 0, s.nr * d_r / np.where(r > 0, r, 1.0), 0.0)
        n = n + np.where(d_m > 0, d_m * cfg.GR_N0 / (lam * rho * np.where(has_g, gq, 1.0)), 0.0)
        n = n + np.where(~cold & (d_l > 0), d_l / _X_SHED, 0.0)
        s.nr = np.maximum(n, 0.0)
    s.qc = c - d_i - d_l
    s.qg = gq + to_g
    s.qr = r + to_r
    s.qv = qv + d_s
    d_ice = (1.0 - a) * (-d_i - d_l) + to_g
    s.theta = s.theta + (-lv_cp * d_s + lf_cp * d_ice) / pi0
    acc = _acc(s)
    tot = np.where(d_i > 0, aut + gaci, 1.0)
    for idx, x in ((2, d_i * aut / tot), (3, d_i * gaci / tot), (4, np.where(cold, d_l, 0.0)),
                   (5, np.where(cold, 0.0, d_l)), (6, d_r), (7, d_m), (8, d_s)):
        acc[:, :, idx] += x.sum(axis=1)


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

    if _two_mom():
        s.nr = _clip_nr(np.maximum(s.qr, 0.0), np.maximum(s.nr, 0.0))   # after advection: mean mass in bounds
        rain_2m(s, g, dt)
    else:
        warm_rain(s, g, dt)
    if thermodynamics._graupel():
        ice_graupel(s, g, dt)
    sedimentation(s, g, dt)

    # Enforce non-negative mixing ratios
    s.qv = np.maximum(s.qv, 0.0)
    s.qc = np.maximum(s.qc, 0.0)
    s.qr = np.maximum(s.qr, 0.0)
    s.qg = np.maximum(s.qg, 0.0)
