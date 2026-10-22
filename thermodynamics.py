"""
Moist thermodynamics with conserved variables θ_l and qt.

Following LES_REBUILD_PLAN.md Phase 1.4, line 268-300:
  "Transport θ_l, qt; diagnose qc from saturation"

Physical definitions:
  θ_l' = liquid water potential temperature PERTURBATION (conserved under phase changes)
  qt   = total water mixing ratio (conserved)

  θ_l' = θ' - (Lv/cp) * qc / π₀     (both are perturbations)
  qt   = qv + qc + qr

IMPORTANT: θ_l is stored as a PERTURBATION (like θ), not absolute value!
  To get absolute temperature: T = (θ_l' + θ0) * π0

Benefits:
  - Conserves total water exactly (no numerical condensation/evaporation)
  - Standard practice in shallow-cumulus LES (DALES, UWLCM, etc.)
  - Eliminates θ/qv/qc consistency problems

References:
  - Stevens et al. (2005) - Evaluation of LES for shallow cumulus
  - Siebesma & Cuijpers (1995) - Parametric assumptions for shallow cumulus
"""

import numpy as np
import config as cfg

try:
    import kernels as _kernels   # numba saturation adjustment (cfg.USE_NUMBA)
except ImportError:
    _kernels = None


def theta_l_from_state(theta: np.ndarray, qc: np.ndarray, pi0: np.ndarray,
                       theta0: np.ndarray = None) -> np.ndarray:
    """
    Compute liquid water potential temperature perturbation from θ' and qc.

    θ_l' = θ' - (Lv/cp) * qc / π₀
    ICE_SCHEME "mixed": θ_l' = θ' - (Lv + (1 - α(T)) Lf)/cp * qc / π₀ (needs theta0 for T)

    Args:
        theta: Potential temperature perturbation θ' [K]
        qc: Cloud water mixing ratio [kg/kg]
        pi0: Exner function π₀ [-] (shape: (1,1,nz))

    Returns:
        θ_l: Liquid water potential temperature PERTURBATION θ_l' [K]
    """
    if _mixed():
        return _theta_l_mixed(theta, qc, theta0, pi0)
    L_cp = cfg.LV / cfg.CP
    return theta - L_cp * qc / pi0


def theta_from_theta_l(theta_l: np.ndarray, qc: np.ndarray, pi0: np.ndarray) -> np.ndarray:
    """
    Recover potential temperature perturbation from θ_l' and qc.

    θ' = θ_l' + (Lv/cp) * qc / π₀

    Args:
        theta_l: Liquid water potential temperature PERTURBATION θ_l' [K]
        qc: Cloud water mixing ratio [kg/kg]
        pi0: Exner function π₀ [-]

    Returns:
        theta: Potential temperature perturbation θ' [K]
    """
    L_cp = cfg.LV / cfg.CP
    return theta_l + L_cp * qc / pi0


def qt_from_state(qv: np.ndarray, qc: np.ndarray, qr: np.ndarray) -> np.ndarray:
    """
    Compute total water mixing ratio.

    qt = qv + qc + qr

    Args:
        qv: Water vapor mixing ratio [kg/kg]
        qc: Cloud water mixing ratio [kg/kg]
        qr: Rain water mixing ratio [kg/kg]

    Returns:
        qt: Total water mixing ratio [kg/kg]
    """
    return qv + qc + qr


def saturation_mixing_ratio(T: np.ndarray, pi0: np.ndarray) -> np.ndarray:
    """
    Saturation water-vapour mixing ratio [kg/kg].
    Uses Buck (1981) approximation for e_s(T) [Pa].

    Args:
        T: Temperature [K]
        pi0: Exner function π₀ [-]

    Returns:
        qs: Saturation mixing ratio [kg/kg]
    """
    T_C = T - 273.15
    e_s = 611.2 * np.exp(17.67 * T_C / (T_C + 243.5))   # Pa
    p   = cfg.P00 * pi0 ** (cfg.CP / cfg.RD)
    return cfg.EPS * e_s / np.maximum(p - e_s, 1.0)


# ── Mixed phase (ICE_SCHEME = "mixed") ─────────────────────────────────────────
# Cloud condensate qc is liquid with fraction α(T) = (T − T_ICE_00)/(T_ICE_0 − T_ICE_00), clipped to
# [0, 1], and ice otherwise. Saturation is over that mix, q_s = α q_sw + (1 − α) q_si (Buck 1981), and
# the latent heat is L_v + (1 − α) L_f, so θ_l is the ice-liquid water potential temperature. The numba
# kernels (kernels.diagnose_cloud_water_mixed / theta_l_mixed) mirror these functions cell by cell.

def _mixed() -> bool:
    return getattr(cfg, 'ICE_SCHEME', 'none') == 'mixed'


def _use_kernels(a: np.ndarray, pi0) -> bool:
    return (_kernels is not None and getattr(cfg, 'USE_NUMBA', True) and a.ndim == 3
            and np.size(pi0) == a.shape[2])


def liquid_fraction(T: np.ndarray):
    """α(T) and dα/dT: liquid fraction of the cloud condensate."""
    t0, t00 = cfg.T_ICE_0, cfg.T_ICE_00
    a = np.clip((T - t00) / (t0 - t00), 0.0, 1.0)
    da = np.where((T > t00) & (T < t0), 1.0 / (t0 - t00), 0.0)
    return a, da


def qsat_mixed(T: np.ndarray, p: np.ndarray):
    """Saturation mixing ratio over the liquid/ice mix and dq_s/dT (Buck 1981 e_w, e_i)."""
    T_C = T - 273.15
    e_w = 611.2 * np.exp(17.67 * T_C / (T_C + 243.5))
    e_i = 611.15 * np.exp(22.452 * T_C / (T_C + 272.55))
    dw = np.maximum(p - e_w, 1.0); di = np.maximum(p - e_i, 1.0)
    q_w = cfg.EPS * e_w / dw; q_i = cfg.EPS * e_i / di
    dq_w = q_w * p / dw * 17.67 * 243.5 / (T_C + 243.5) ** 2
    dq_i = q_i * p / di * 22.452 * 272.55 / (T_C + 272.55) ** 2
    a, da = liquid_fraction(T)
    return a * q_w + (1 - a) * q_i, a * dq_w + (1 - a) * dq_i + da * (q_w - q_i)


def _theta_l_mixed(theta, qc, theta0, pi0):
    lv_cp, lf_cp = cfg.LV / cfg.CP, cfg.LF / cfg.CP
    if _use_kernels(theta, pi0):
        return _kernels.theta_l_mixed(
            np.ascontiguousarray(theta, dtype=np.float64), np.ascontiguousarray(qc, dtype=np.float64),
            np.asarray(theta0, dtype=np.float64), np.asarray(pi0, dtype=np.float64).reshape(-1),
            lv_cp, lf_cp, cfg.T_ICE_0, cfg.T_ICE_00)
    th0 = np.asarray(theta0).reshape(1, 1, -1)
    a, _ = liquid_fraction((theta + th0) * pi0)
    return theta - (lv_cp + (1 - a) * lf_cp) * qc / pi0


def diagnose_cloud_water_mixed(theta_l: np.ndarray, qt: np.ndarray,
                               theta0: np.ndarray, pi0: np.ndarray) -> tuple:
    """
    Mixed-phase saturation adjustment: (θ_l', qt) → (qv, qc, θ').

    Solves T = T_l + (L_v + (1 − α(T)) L_f)/c_p · max(qt − q_s(T), 0) for T with a safeguarded
    Newton iteration on the bracket [T_l, T_l + (L_v + L_f) qt / c_p] (the residual is monotone in T).
    """
    lv_cp, lf_cp = cfg.LV / cfg.CP, cfg.LF / cfg.CP
    pi0 = np.asarray(pi0, dtype=np.float64)
    if _use_kernels(theta_l, pi0):
        pi0_1d = np.ascontiguousarray(pi0.reshape(-1))
        p = cfg.P00 * pi0_1d ** (cfg.CP / cfg.RD)
        qv, qc, theta, _ = _kernels.diagnose_cloud_water_mixed(
            np.ascontiguousarray(theta_l, dtype=np.float64), np.ascontiguousarray(qt, dtype=np.float64),
            np.asarray(theta0, dtype=np.float64), pi0_1d, p, lv_cp, lf_cp, cfg.EPS,
            cfg.T_ICE_0, cfg.T_ICE_00)
        return qv, qc, theta

    th0 = np.asarray(theta0).reshape(1, 1, -1)
    p = cfg.P00 * pi0 ** (cfg.CP / cfg.RD) * np.ones_like(theta_l)
    Tl = (theta_l + th0) * pi0
    qs, _ = qsat_mixed(Tl, p)
    sat = qt > qs
    T = Tl.copy()
    if sat.any():
        Ts, Tls, q, ps = Tl[sat], Tl[sat], qt[sat], p[sat]
        lo, hi = Tls.copy(), Tls + (lv_cp + lf_cp) * q
        done = np.zeros(Ts.shape, dtype=bool)
        for _ in range(30):
            qs, dqs = qsat_mixed(Ts, ps)
            a, da = liquid_fraction(Ts)
            c = np.maximum(q - qs, 0.0)
            L = lv_cp + (1 - a) * lf_cp
            R = Ts - Tls - L * c
            done |= np.abs(R) < 1e-9          # converged cells stay put (as the per-cell kernel's break)
            if done.all():
                break
            hi = np.where(R > 0, Ts, hi)
            lo = np.where(R > 0, lo, Ts)
            dR = 1.0 + lf_cp * da * c + np.where(c > 0, L * dqs, 0.0)
            Tn = Ts - R / dR
            Ts = np.where(done, Ts, np.where((Tn > lo) & (Tn < hi), Tn, 0.5 * (lo + hi)))
        T[sat] = Ts
    qs, _ = qsat_mixed(T, p)
    qc = np.where(sat, np.maximum(qt - qs, 0.0), 0.0)
    a, _ = liquid_fraction(T)
    theta = theta_l + (lv_cp + (1 - a) * lf_cp) * qc / pi0
    return np.maximum(qt - qc, 0.0), qc, theta


def diagnose_cloud_water(theta_l: np.ndarray, qt: np.ndarray,
                         theta0: np.ndarray, pi0: np.ndarray,
                         max_iterations: int = 5) -> tuple:
    """
    Diagnose qc, qv from conserved variables θ_l', qt via saturation adjustment.

    This is the core of the θ_l/qt framework: given conserved (θ_l', qt),
    find the partition into (qv, qc) that satisfies saturation.

    Algorithm:
    1. Guess: qc = 0 (unsaturated)
    2. Compute θ' from θ_l', then T from θ'+θ0
    3. Check if saturated: qt > qs(T)?
    4. If yes, iterate with Newton-Raphson to find qc

    Args:
        theta_l: Liquid water potential temperature PERTURBATION θ_l' [K]
        qt: Total water mixing ratio [kg/kg]
        theta0: Background potential temperature θ0 [K] (1D array, nz)
        pi0: Exner function π₀ [-] (shape (1,1,nz))
        max_iterations: Maximum Newton iterations

    Returns:
        qv: Water vapor [kg/kg]
        qc: Cloud water [kg/kg]
    """
    if (_kernels is not None and getattr(cfg, 'USE_NUMBA', True) and theta_l.ndim == 3
            and np.size(pi0) == theta_l.shape[2] and np.size(theta0) == theta_l.shape[2]):
        pi0_1d = np.ascontiguousarray(np.asarray(pi0, dtype=np.float64).reshape(-1))
        p = cfg.P00 * pi0_1d ** (cfg.CP / cfg.RD)
        qv, qc, any_sat = _kernels.diagnose_cloud_water(
            np.ascontiguousarray(theta_l), np.ascontiguousarray(qt), np.asarray(theta0, dtype=np.float64),
            pi0_1d, p, cfg.LV / cfg.CP, cfg.LV, cfg.RV, cfg.EPS, max_iterations)
        return (qv, qc) if any_sat else (qt.copy(), qc)

    # Initial guess: all vapor, no condensate
    qc = np.zeros_like(theta_l)
    qv = qt.copy()

    # Compute temperature from initial guess
    theta_abs = theta_l + theta0[np.newaxis, np.newaxis, :]
    T = theta_abs * pi0
    qs = saturation_mixing_ratio(T, pi0)

    # Check for supersaturation
    supersaturated = qt > qs

    if not supersaturated.any():
        # Entire domain is unsaturated
        return qv, qc

    # Newton-Raphson iteration for saturated regions
    # We need to find qc such that: qv = qs(T(qc))
    # where qv = qt - qc and T depends on qc through θ = θ_l + (Lv/cp)*qc/π₀

    L_cp = cfg.LV / cfg.CP

    for iteration in range(max_iterations):
        # Current temperature (depends on current qc guess)
        theta_abs = theta_l + L_cp * qc / pi0 + theta0[np.newaxis, np.newaxis, :]
        T = theta_abs * pi0
        qs = saturation_mixing_ratio(T, pi0)

        # Residual: how much we're supersaturated
        residual = qt - qc - qs

        # Only update supersaturated regions
        update_mask = supersaturated & (np.abs(residual) > 1e-8)

        if not update_mask.any():
            break  # Converged

        # Newton step: dqs/dT from Clausius-Clapeyron
        dqsdT = qs * cfg.LV / (cfg.RV * T**2)

        # Temperature changes with qc:
        # Since Δθ = (L_v/c_p) * Δq_c / π₀, we have ΔT = π₀ * Δθ = (L_v/c_p) * Δq_c
        # Therefore: dT/dqc = L_v/c_p (NO π₀!)
        dTdqc = L_cp

        # Chain rule: dqs/dqc = (dqs/dT) * (dT/dqc)
        dqsdqc = dqsdT * dTdqc

        # Newton update: residual / derivative
        # Residual = qt - qc - qs(qc)
        # d(Residual)/dqc = -1 - dqs/dqc
        denominator = 1.0 + dqsdqc
        delta_qc = residual / denominator

        # Update qc (only in supersaturated regions)
        qc = np.where(update_mask, qc + delta_qc, qc)
        qc = np.maximum(qc, 0.0)  # Non-negative
        qc = np.minimum(qc, qt)    # Can't exceed total water

    # Final partition
    qv = qt - qc
    qv = np.maximum(qv, 0.0)  # Non-negative

    return qv, qc


def convert_to_theta_l_qt(theta: np.ndarray, qv: np.ndarray, qc: np.ndarray, qr: np.ndarray,
                          pi0: np.ndarray, theta0: np.ndarray = None) -> tuple:
    """
    Convert from (θ', qv, qc, qr) to (θ_l', qt, qr).

    This is used for initialization when transitioning from old framework.

    Args:
        theta: Potential temperature perturbation θ' [K]
        qv: Water vapor [kg/kg]
        qc: Cloud water [kg/kg]
        qr: Rain water [kg/kg]
        pi0: Exner function [-]

    Returns:
        theta_l: Liquid water potential temperature PERTURBATION θ_l' [K]
        qt: Total water [kg/kg]
        qr: Rain water [kg/kg] (unchanged)
    """
    theta_l = theta_l_from_state(theta, qc, pi0, theta0)
    qt = qt_from_state(qv, qc, qr)
    return theta_l, qt, qr


def convert_from_theta_l_qt(theta_l: np.ndarray, qt: np.ndarray, qr: np.ndarray,
                            theta0: np.ndarray, pi0: np.ndarray) -> tuple:
    """
    Convert from (θ_l', qt, qr) to (θ', qv, qc, qr).

    This is used for output/diagnostics.

    Args:
        theta_l: Liquid water potential temperature PERTURBATION θ_l' [K]
        qt: Total water [kg/kg]
        qr: Rain water [kg/kg]
        theta0: Background potential temperature θ0 [K] (1D array, nz)
        pi0: Exner function [-]

    Returns:
        theta: Potential temperature perturbation θ' [K]
        qv: Water vapor [kg/kg]
        qc: Cloud water [kg/kg]
        qr: Rain water [kg/kg] (unchanged)
    """
    # Remove rain from total water (rain doesn't contribute to saturation calculation)
    qt_norain = qt - qr
    qt_norain = np.maximum(qt_norain, 0.0)

    if _mixed():
        qv, qc, theta = diagnose_cloud_water_mixed(theta_l, qt_norain, theta0, pi0)
        return theta, qv, qc, qr

    # Diagnose vapor/cloud partition
    qv, qc = diagnose_cloud_water(theta_l, qt_norain, theta0, pi0)

    # Recover potential temperature perturbation
    theta = theta_from_theta_l(theta_l, qc, pi0)

    return theta, qv, qc, qr
