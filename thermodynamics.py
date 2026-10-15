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


def theta_l_from_state(theta: np.ndarray, qc: np.ndarray, pi0: np.ndarray) -> np.ndarray:
    """
    Compute liquid water potential temperature perturbation from θ' and qc.

    θ_l' = θ' - (Lv/cp) * qc / π₀

    Args:
        theta: Potential temperature perturbation θ' [K]
        qc: Cloud water mixing ratio [kg/kg]
        pi0: Exner function π₀ [-] (shape: (1,1,nz))

    Returns:
        θ_l: Liquid water potential temperature PERTURBATION θ_l' [K]
    """
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
                          pi0: np.ndarray) -> tuple:
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
    theta_l = theta_l_from_state(theta, qc, pi0)
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

    # Diagnose vapor/cloud partition
    qv, qc = diagnose_cloud_water(theta_l, qt_norain, theta0, pi0)

    # Recover potential temperature perturbation
    theta = theta_from_theta_l(theta_l, qc, pi0)

    return theta, qv, qc, qr
