"""
Continental shallow-cumulus sounding ("land_cu").

Layers (z = height above surface):
  0–700 m      mixed layer:  θ = 300 K, qv constant with RH = 0.93 at 700 m
  700–1700 m   cloud layer:  dθ/dz = 4 K/km, qv linear down to RH = 0.80 at 1700 m
  1700–1900 m  inversion:    θ +3 K across the layer, RH linear 0.80 → 0.40
  > 1900 m     free trop.:   dθ/dz = 5 K/km, RH = 0.40

Moisture is set from the model's own q_sat (thermodynamics.saturation_mixing_ratio)
evaluated on the reference state, so RH targets are exact on the model grid.

Settings applied by configure(): see that function (Tasks 6–9).

Initial perturbations: uniform white noise θ' ∈ [-0.1, 0.1] K and
qv' ∈ [-0.1, 0.1] g/kg in cells below 200 m only.
"""

import numpy as np
import config as cfg
from thermodynamics import saturation_mixing_ratio

THETA_ML = 300.0       # K, mixed-layer θ
Z_ML     = 700.0       # m, mixed-layer top (cloud base)
Z_CL     = 1700.0      # m, cloud-layer top (inversion base)
Z_INV    = 1900.0      # m, inversion top
GAMMA_CL = 4.0e-3      # K/m, cloud-layer dθ/dz
DTHETA_INV = 3.0       # K, jump across inversion
GAMMA_FT = 5.0e-3      # K/m, free-troposphere dθ/dz

RH_ML_TOP = 0.93       # RH at mixed-layer top
RH_CL_TOP = 0.80       # RH at cloud-layer top
RH_FT     = 0.40       # RH at inversion top and above

Z_NOISE     = 200.0    # m, perturbations only below this height
THETA_NOISE = 0.1      # K
QV_NOISE    = 1.0e-4   # kg/kg (0.1 g/kg)


def u_geo_profile(z: np.ndarray) -> np.ndarray:
    """Geostrophic (and initial) wind u_g(z) = 5 + 4·min(z/3000, 1) m/s."""
    return 5.0 + 4.0 * np.minimum(z / 3000.0, 1.0)


def configure() -> None:
    """
    Apply land_cu settings to config (forcing, wind, grid, sponge, time step).

    Called by Simulation when CASE == "land_cu", before the grid is built.
    Applied once: a run script may call configure() itself and then override values.
    """
    if getattr(cfg, '_LAND_CU_CONFIGURED', False):
        return
    cfg._LAND_CU_CONFIGURED = True

    # Domain: 6.4 × 6.4 km at 100 m (was set by run_land_cu.py before run_case.py)
    cfg.NX = 64
    cfg.NY = 64

    # Surface fluxes: homogeneous, ramped over 45 min, no shading
    cfg.SHF = 150.0
    cfg.LHF = 300.0
    cfg.FLUX_RAMP_TIME = 2700.0
    cfg.SHF_SIGMA = 0.0
    cfg.ENABLE_FLUX_PATTERN = False
    cfg.ENABLE_CLOUD_SHADING = False

    # Sheared wind: u = u_g(z), v = 0
    cfg.U_GEO_PROFILE = u_geo_profile
    cfg.U_GEO = 5.0
    cfg.V_GEO = 0.0

    # Large-scale forcing
    cfg.Z_BL = Z_ML                      # mixed-layer top (BL diagnostics)
    cfg.NUDGE_Z_TOP = Z_INV              # no nudging below the inversion top
    cfg.TAU_NUDGE_T = 6 * 3600.0
    cfg.TAU_NUDGE_Q = 6 * 3600.0
    cfg.W_SUBS = 0.004                   # w_ls: 0 → −4 mm/s at 1900 m → 0 at 3000 m
    cfg.SUBS_Z_MAX = Z_INV
    cfg.SUBS_Z_ZERO = 3000.0
    cfg.SUBS_GRADIENT = "mean"
    cfg.RAD_COOLING = -2.0               # K/day below 1900 m, tapering to 0 at 2500 m
    cfg.RAD_Z_TOP = Z_INV
    cfg.RAD_Z_ZERO = 2500.0

    # Vertical grid: 40 m to 3000 m, stretched to 4000 m; sponge from 3200 m
    cfg.GRID_TYPE = "uniform_stretched"
    cfg.DZ_UNIFORM = 40.0
    cfg.Z_UNIFORM_TOP = 3000.0
    cfg.Z_TOP = 4000.0
    cfg.N_STRETCH = 12
    cfg.Z_SPONGE = 3200.0

    # Adaptive dt on the summed 3-D Courant number, capped at 5 s
    cfg.CFL_SUM = True
    cfg.CFL_MAX = 0.8
    cfg.DT = 5.0


def theta_profile(z: np.ndarray) -> np.ndarray:
    """Piecewise-linear θ₀(z) [K]."""
    theta_cl_top  = THETA_ML + GAMMA_CL * (Z_CL - Z_ML)
    theta_inv_top = theta_cl_top + DTHETA_INV
    return np.select(
        [z <= Z_ML, z <= Z_CL, z <= Z_INV],
        [np.full_like(z, THETA_ML),
         THETA_ML + GAMMA_CL * (z - Z_ML),
         theta_cl_top + DTHETA_INV * (z - Z_CL) / (Z_INV - Z_CL)],
        theta_inv_top + GAMMA_FT * (z - Z_INV),
    )


def exner_profile(theta0: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Hydrostatic π₀(z), integrated exactly as in State._build_reference / State._pi0."""
    pi0 = np.ones(len(z))
    for k in range(1, len(z)):
        dz = z[k] - z[k - 1]
        theta_mid = 0.5 * (theta0[k] + theta0[k - 1])
        pi0[k] = pi0[k - 1] - cfg.G * dz / (cfg.CP * theta_mid)
    return pi0


def qsat_profile(theta0: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Model q_sat [kg/kg] on the reference state."""
    pi0 = exner_profile(theta0, z)
    return saturation_mixing_ratio(theta0 * pi0, pi0)


def reference_profiles(z: np.ndarray):
    """
    Return (θ₀, qv₀) on cell centres z.

    The anchor points (700, 1700, 1900 m) generally fall between grid levels,
    so q_sat there is interpolated from a fine auxiliary column with the same
    surface reference (z[0]).
    """
    theta0 = theta_profile(z)
    qsat = qsat_profile(theta0, z)

    # q_sat at the anchor heights, from a fine column starting at z[0]
    z_fine = np.arange(z[0], max(z[-1], Z_INV) + 1.0, 1.0)
    qsat_fine = qsat_profile(theta_profile(z_fine), z_fine)
    qsat_at = lambda zz: np.interp(zz, z_fine, qsat_fine)

    qv_ml = RH_ML_TOP * qsat_at(Z_ML)
    qv_cl_top = RH_CL_TOP * qsat_at(Z_CL)

    rh_inv = RH_CL_TOP + (RH_FT - RH_CL_TOP) * (z - Z_CL) / (Z_INV - Z_CL)
    qv0 = np.select(
        [z <= Z_ML, z <= Z_CL, z <= Z_INV],
        [np.full_like(z, qv_ml),
         qv_ml + (qv_cl_top - qv_ml) * (z - Z_ML) / (Z_CL - Z_ML),
         rh_inv * qsat],
        RH_FT * qsat,
    )
    return theta0, qv0


def init_perturbations(s, g) -> None:
    """Background qv plus white-noise θ', qv' in the lowest 200 m."""
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    shape = (g.nx, g.ny, g.nz)
    low = (g.z < Z_NOISE)[np.newaxis, np.newaxis, :]

    s.qv[:] = s.qv0[np.newaxis, np.newaxis, :]
    s.theta += THETA_NOISE * rng.uniform(-1.0, 1.0, shape) * low
    s.qv    += QV_NOISE    * rng.uniform(-1.0, 1.0, shape) * low


def print_rh_profile(s, g) -> None:
    """Print the initial RH profile (horizontal mean and max) and qc max."""
    pi0 = s._pi0()
    T = (s.theta + s.theta0[np.newaxis, np.newaxis, :]) * pi0
    rh = s.qv / saturation_mixing_ratio(T, pi0)
    rh_mean = rh.mean(axis=(0, 1))
    rh_max = rh.max(axis=(0, 1))
    qc_max = s.qc.max(axis=(0, 1))

    print("=" * 66)
    print("land_cu initial sounding")
    print("=" * 66)
    print(f"{'z (m)':>8} {'θ0 (K)':>9} {'qv0 (g/kg)':>11} {'RH mean':>9} {'RH max':>8} {'qc max':>8}")
    for k in range(g.nz):
        if g.z[k] > Z_INV + 600.0 and k % 5:
            continue   # thin out the free troposphere
        print(f"{g.z[k]:8.0f} {s.theta0[k]:9.2f} {s.qv0[k]*1e3:11.3f} "
              f"{rh_mean[k]:9.3f} {rh_max[k]:8.3f} {qc_max[k]:8.1e}")
    print(f"Max RH anywhere: {rh.max():.3f}   Max qc anywhere: {s.qc.max():.1e} kg/kg")
    print("=" * 66)
